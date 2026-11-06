"""Policy fitting, validation-only OOB selection and held-out evaluation."""
import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from policy_core.config import SIMULATION_PROFILE
from policy_core.data import prepare_policy_data
from policy_core.learning import fit_candidate
from policy_core.models import CategoricalSievePolicy
from policy_core.oob import coefficient_grid, select_oob, validation_scores
from policy_core.robust_values import robust_values
from .dgp import (CASES, DESIGN, fitting_seeds, held_out_data, oob_counts,
                  shift_direction_seeds, shift_resampling_seed, training_data)
from . import shift

WIDTHS = tuple(range(2, 257, 2))
METHODS = ('robust', 'non-robust')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit(study, case, rep, width, method, destination, device, data=None):
    """Fit one candidate once; an existing candidate directory is never overwritten."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if data is None:
        data = training_data(study, case, rep)
    args = SimpleNamespace(**DESIGN['training_options'])
    args.policy_hidden_width = width
    args.robust_met = method
    args.model_dir = destination
    # The first 500 validation rows precede the 4500 fitting rows in the API.
    learning = {k:np.concatenate([data['master_validation'][k][:500],
                                  data['master_train'][k][:4500]], axis=0)
                for k in data['master_train']}
    preparation, candidate = fitting_seeds(study, case, rep, width)
    set_seed(preparation)
    policy_data = prepare_policy_data(learning, data['test'], args, limit_test=2500,
        alpha_bound_order=SIMULATION_PROFILE.alpha_bound_order, validation_fraction=.1)
    set_seed(candidate)
    path = destination/'model.pt'
    _, validation, alphas, value, history = fit_candidate(policy_data, 2,
        rep, path, SIMULATION_PROFILE,
        torch.device(device), return_diagnostics=True)
    save(destination/'complete.json', dict(study=study, case=case, rep=rep, width=width,
        method=method, preparation_seed=preparation, candidate_seed=candidate,
        validation_value=float(validation), training_value=float(value),
        alphas=[float(a) for a in alphas], history=history))
    return path


def predict(paths, widths, x, device):
    tensor = torch.tensor(x, dtype=torch.float32, device=device)
    predictions = []
    for path, width in zip(paths, widths):
        policy = CategoricalSievePolicy(4, 2, 2, torch.device(device), width=width)
        policy.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        policy.eval()
        with torch.no_grad():
            predictions.append(torch.softmax(policy(tensor), 1).argmax(1).cpu().numpy())
    return np.column_stack(predictions)


def select(study, case, rep, method, validation, decisions):
    eta = .2 if method == 'robust' else 0.
    counts = oob_counts(study, case, rep)
    scores = validation_scores(validation, decisions, counts, eta=eta)
    chosen, curve = select_oob(**scores, widths=WIDTHS, coefficients=coefficient_grid(),
                               cap=128 if study == 'baseline' else 256)
    return chosen, dict(**scores, counts=counts, curve=curve)


def baseline_evaluation(test, decisions, selected):
    values = robust_values(test['Y'], test['T'], decisions, test['prop_score'], eta=.2)['value'][0]
    q = float(values[selected['width']//2-1])
    oracle = float(values[:64].max())
    return dict(test_value=q, oracle_value=oracle, test_regret=oracle-q)


def shift_evaluation(rep, test, predictions):
    """Minimum policy mean over 100 paired environments at each severity."""
    n = len(test['X'])
    y = test['Y_all'].astype(np.float64)
    p0, u, z, risky = (test[k].ravel() for k in
                       ('tail_p0', 'tail_u', 'tail_z', 'tail_risky_action'))
    baseline = {method:y[np.arange(n), pred] for method, pred in predictions.items()}
    h = shift.susceptibility(test['X'], shift_direction_seeds(rep))
    rows = []
    for severity in shift.SEVERITIES:
        weights, _, _ = shift.tilt_weights(test['X'], severity)
        for m in range(100):
            if severity == 0:
                cov = con = {method:float(values.mean()) for method, values in baseline.items()}
            else:
                rng = np.random.default_rng(shift_resampling_seed(rep, severity, m))
                indices = rng.choice(n, n, replace=True, p=weights)
                cov = {method:float(values[indices].mean()) for method, values in baseline.items()}
                p, _, _ = shift.solve_concept_probability(p0, h[m], severity)
                potential = shift.concept_outcomes(risky, u, z, p).astype(np.float32).astype(np.float64)
                con = {method:float(potential[np.arange(n), pred].mean())
                       for method, pred in predictions.items()}
            for mode, values in (('Covariate shift', cov), ('Concept drift', con)):
                rows.append(dict(rep=rep, mode=mode, severity=severity, environment=m, **values))
    return pd.DataFrame(rows)


def summarize_baseline(frame):
    return frame.groupby('case').agg(n=('rep','size'), median_width=('width','median'),
        mean_q=('test_value','mean'), sd_q=('test_value','std'),
        mean_regret=('test_regret','mean'), sd_regret=('test_regret','std')).reset_index()


def summarize_shift(frame):
    rows = []
    for (mode, severity), part in frame.groupby(['mode','severity']):
        n = len(part)
        a, b = part['robust'], part['non-robust']
        rows.append(dict(mode=mode, severity=severity, n=n, robust_mean=a.mean(),
            robust_se=a.std(ddof=1)/np.sqrt(n), nonrobust_mean=b.mean(),
            nonrobust_se=b.std(ddof=1)/np.sqrt(n), improvement_percent=100*(a-b).mean()/b.mean()))
    return pd.DataFrame(rows)


def run(study, cases, reps, output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output/'plan.json',dict(study=study,cases=cases,reps=reps,widths=WIDTHS,device=device))
    methods = ('robust',) if study == 'baseline' else METHODS
    selections, selected_paths = [], {}
    for case in cases:
        for rep in reps:
            data = training_data(study, case, rep)
            validation = (held_out_data(case, rep, 'validation') if study == 'baseline'
                          else {k:v[:500] for k,v in data['master_validation'].items()})
            for method in methods:
                pool = output/f'{case}/rep_{rep:03d}/{method}'
                paths = []
                for width in WIDTHS:
                    paths.append(fit(study, case, rep, width, method, pool/f'w{width}', device, data))
                    print(f'{case} rep={rep} {method} width={width} fitted', flush=True)
                decisions = predict(paths, WIDTHS, validation['X'], device)
                pick, scores = select(study, case, rep, method, validation, decisions)
                pool.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(pool/'calibration.npz', decisions=decisions, **scores)
                row = dict(case=case, rep=rep, method=method, **pick)
                selections.append(row)
                selected_paths[case, rep, method] = paths
    # All choices are fixed on validation before held-out policy evaluation.
    pd.DataFrame(selections).to_csv(output/'selections_frozen.csv',index=False)
    if study == 'baseline':
        rows = []
        for chosen in selections:
            case, rep, method = chosen['case'], chosen['rep'], chosen['method']
            test = held_out_data(case, rep, 'test')
            pred = predict(selected_paths[case, rep, method], WIDTHS, test['X'], device)
            rows.append(dict(**chosen, **baseline_evaluation(test, pred, chosen)))
        frame = pd.DataFrame(rows)
        summary = summarize_baseline(frame)
    else:
        environments = []
        for rep in reps:
            test = {k:v[:2500] for k,v in training_data(study, 'moderate', rep)['test'].items()}
            predictions = {}
            for method in methods:
                pick = next(r for r in selections if r['rep']==rep and r['method']==method)
                path = selected_paths['moderate', rep, method][pick['width']//2-1]
                predictions[method] = predict([path],[pick['width']],test['X'],device)[:,0]
            environments.append(shift_evaluation(rep, test, predictions))
        environments = pd.concat(environments,ignore_index=True)
        environments.to_csv(output/'environments.csv',index=False)
        frame = environments.groupby(['rep','mode','severity'])[list(METHODS)].min().reset_index()
        summary = summarize_shift(frame)
    frame.to_csv(output/'replication_results.csv',index=False)
    summary.to_csv(output/'summary.csv',index=False)
    save(output/'complete.json',dict(study=study,replications=len(reps),cases=cases))
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['fit','run'])
    parser.add_argument('--study',choices=['baseline','shift'],required=True)
    parser.add_argument('--cases',nargs='+',choices=CASES)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--reps',nargs='+',type=int,default=[0])
    group.add_argument('--all-reps',action='store_true')
    parser.add_argument('--width',type=int,choices=WIDTHS)
    parser.add_argument('--method',choices=METHODS,default='robust')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    cases = args.cases or (['moderate'] if args.study=='shift' else list(CASES))
    reps = list(range(100)) if args.all_reps else args.reps
    if len(set(reps))!=len(reps) or not all(0<=r<100 for r in reps):
        parser.error('Replication IDs must be distinct integers in 0,...,99')
    if len(set(cases))!=len(cases): parser.error('Cases must be distinct')
    if args.study=='shift' and cases!=['moderate']:
        parser.error('The shift experiment uses --cases moderate')
    if args.study=='baseline' and args.method!='robust':
        parser.error('The baseline PDRP experiment uses robust candidate training')
    torch.set_num_threads(1)
    if args.command=='fit':
        if len(reps)!=1 or len(cases)!=1 or args.width is None:
            parser.error('fit requires one case, one replication and --width')
        fit(args.study,cases[0],reps[0],args.width,args.method,args.output,args.device)
    else:
        if args.width is not None: parser.error('run uses the full fixed width grid')
        run(args.study,cases,reps,args.output,args.device)
