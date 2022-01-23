#!/usr/bin/env python

from subprocess import Popen, PIPE, CalledProcessError

# 'gpt2'
models = ['albert-base-v2']
# models = ['bert-base-uncased']
percents = '5,10,15,20'
# percents = '5'
epochs = '5,7,10'
# epochs = '2'

script_command = ['python', 'scripts/train.py']

for model in models:
    script_params = (f'--model {model} '
                     f'--epochs {epochs} --percents {percents} '
                     f'--output output/{model}.json').split(' ')

    whole_params = script_command + script_params
    # subprocess.run(whole_params, capture_output=True)

    with Popen(whole_params, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')  # process line here

    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)
