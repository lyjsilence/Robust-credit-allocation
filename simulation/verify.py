"""Compare freshly produced outputs with the published numerical references."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .experiment import summarize_baseline, summarize_shift

REFERENCE = Path(__file__).resolve().parents[1]/'reference_results.json'


def state_digest(path):
    """Hash named tensor contents, independent of torch.save container metadata."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    digest = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().cpu().contiguous().numpy()
        identity = json.dumps([name, str(array.dtype), list(array.shape)], separators=(',', ':'))
        digest.update(identity.encode())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def compare_rows(actual, expected, keys, columns):
    actual = pd.DataFrame(actual)
    expected = pd.DataFrame(expected)
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise AssertionError('Duplicate comparison identities')
    joined = actual.merge(expected, on=keys, suffixes=('', '_expected'), validate='one_to_one')
    if len(joined)!=len(actual) or len(actual)==0:
        raise AssertionError('No matching reference for one or more rows')
    maximum = 0.
    for col in columns:
        a, b = joined[col].to_numpy(float), joined[col+'_expected'].to_numpy(float)
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,err_msg=col)
        maximum = max(maximum,float(np.max(np.abs(a-b))))
    return maximum


def verify_fit(directory, reference):
    info = json.loads((directory/'complete.json').read_text(encoding='utf-8'))
    keys = ('study','case','rep','width','method')
    match = [r for r in reference['fit_checks'] if all(r[k]==info[k] for k in keys)]
    if len(match)!=1:
        raise ValueError('No tensor reference for this diagnostic fit; see the five README examples')
    actual = state_digest(directory/'model.pt')
    if actual!=match[0]['state_sha256']:
        raise AssertionError('Model tensor contents differ from the reference')
    print('PASS: model tensor contents match exactly')


def verify_run(directory, reference):
    complete = json.loads((directory/'complete.json').read_text(encoding='utf-8'))
    plan = json.loads((directory/'plan.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(directory/'replication_results.csv',float_precision='round_trip')
    picks = pd.read_csv(directory/'selections_frozen.csv',float_precision='round_trip')
    if complete['study']!=plan['study']:
        raise AssertionError('Plan and completion record disagree')
    error = 0.
    if complete['study']=='baseline':
        expected_rows = len(plan['cases'])*len(plan['reps'])
        if len(frame)!=expected_rows or len(picks)!=expected_rows:
            raise AssertionError('Incomplete baseline run')
        if 0 in plan['reps']:
            error=compare_rows(frame[frame.rep==0],reference['baseline_rep0'],['case','rep'],
                               ['width','coefficient','test_value','oracle_value','test_regret'])
        if sorted(plan['reps'])==list(range(100)):
            error=max(error,compare_rows(summarize_baseline(frame),reference['baseline_100'],['case'],
                ['n','median_width','mean_q','sd_q','mean_regret','sd_regret']))
    else:
        if len(frame)!=14*len(plan['reps']) or len(picks)!=2*len(plan['reps']):
            raise AssertionError('Incomplete perturbation run')
        if 0 in plan['reps']:
            error=compare_rows(frame[frame.rep==0],reference['shift_rep0'],['rep','mode','severity'],
                               ['robust','non-robust'])
            error=max(error,compare_rows(picks[picks.rep==0],reference['shift_rep0_picks'],['method'],
                                        ['width','coefficient']))
        if sorted(plan['reps'])==list(range(100)):
            error=max(error,compare_rows(summarize_shift(frame),reference['shift_100'],['mode','severity'],
                ['n','robust_mean','robust_se','nonrobust_mean','nonrobust_se','improvement_percent']))
    if 0 not in plan['reps'] and sorted(plan['reps'])!=list(range(100)):
        raise ValueError('This subset has no bundled reference; use replication 0 or all 100')
    print(f'PASS: available replication/table references match; maximum error {error:.3g}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    option=parser.add_mutually_exclusive_group(required=True)
    option.add_argument('--fit',type=Path)
    option.add_argument('--run',type=Path)
    args=parser.parse_args()
    reference=json.loads(REFERENCE.read_text(encoding='utf-8'))
    if args.fit:
        verify_fit(args.fit,reference)
    else:
        verify_run(args.run,reference)


if __name__=='__main__':
    main()
