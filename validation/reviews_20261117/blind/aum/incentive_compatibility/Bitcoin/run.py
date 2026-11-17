import matlab.engine
import json
import logging

logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p', level=logging.INFO)

logging.info('Start')

output_file = 'output.json'
try:
    with open(output_file) as f:
        output_data = json.load(f)
except FileNotFoundError:
    output_data = {}

maxForkLen = 30.0

for gamma in [0.0, 0.5, 1.0]:
    for alpha in [0.25, 0.3, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45]:
        logging.info(f'alpha: {alpha}, gamma: {gamma}, maxForkLen: {maxForkLen}')

        param_string = f'{alpha}_{gamma}_{maxForkLen}'

        if param_string not in output_data.keys():
            output_data[param_string] = {}

            eng = matlab.engine.start_matlab()
            r = eng.Init(alpha, maxForkLen, gamma, nargout=1)
            output_data[param_string] = r
            eng.quit()
            logging.info(f'alpha: {alpha}, result: {r}')
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=4)
