'''The shipped models: the hashes of their weights, and their cards'''
import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path

L = importlib.import_module('CATCH.Localizer')
ROOT = Path(__file__).resolve().parent.parent
KINDS = ('localizer', 'estimator')


def shipped(kind: str) -> Path:
    return ROOT / 'models' / kind / 'default'


class TestWeightsDigest(unittest.TestCase):

    def test_sha256_of_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'weights.pt'
            path.write_bytes(b'abc')
            self.assertEqual(
                L.weights_digest(path),
                'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f2'
                '0015ad')
            # a file longer than the block that is read at a time
            data = bytes(range(256)) * (3 << 12)
            path.write_bytes(data)
            self.assertEqual(L.weights_digest(str(path)),
                             hashlib.sha256(data).hexdigest())


class TestShippedModels(unittest.TestCase):

    def test_the_recorded_hash_is_that_of_the_weights(self) -> None:
        for kind in KINDS:
            config = json.loads((shipped(kind) / 'configuration.json')
                                .read_text())
            self.assertEqual(config['weights_sha256'],
                             L.weights_digest(shipped(kind) / 'weights.pt'),
                             f'{kind}: the weights changed; record the new '
                             'hash in the configuration and the card')

    def test_each_model_has_a_card_for_its_weights(self) -> None:
        for kind in KINDS:
            card = (shipped(kind) / 'MODEL_CARD.md').read_text()
            digest = L.weights_digest(shipped(kind) / 'weights.pt')
            self.assertIn(digest, card, f'{kind}: the card names other '
                          'weights')
            size = (shipped(kind) / 'weights.pt').stat().st_size
            self.assertIn(f'{size:,} bytes', card)
            for heading in ('## Intended use', '## Training data',
                            '## Evaluation', '## Known limitations'):
                self.assertIn(heading, card, kind)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
