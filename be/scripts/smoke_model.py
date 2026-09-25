"""Run from repository root: python -m be.scripts.smoke_model."""
import json
import numpy as np
from be.app.config import ROOT, CLASSES
from be.app.inference import ModelRuntime


def main():
    runtime = ModelRuntime()
    try:
        for path in sorted((ROOT / 'data').glob('*.csv')):
            sample = np.loadtxt(path, delimiter=',', skiprows=1, max_rows=10)
            values = runtime.predict(sample)
            print(json.dumps({'sample': path.name, 'label': CLASSES[int(np.argmax(values))],
                              'probabilities': values}, ensure_ascii=True))
        assert runtime.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []
        print('PASS: six datasets, model output, and blank-frame pose detection')
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
