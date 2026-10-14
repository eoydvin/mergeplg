# mergeplg

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- SPHINX-START -->

`mergeplg` is a collection of methods to merge rainfall observations from
sensors with different geometries into a single rainfall field. Typically the
sensors are rain gauges (point), commercial microwave links / CML (line) and
weather radar or satellites (grid).

## Features

- **Interpolation** of point (gauge) and line (CML) observations onto a grid
  using [IDW](https://en.wikipedia.org/wiki/Inverse_distance_weighting) or
  [kriging](https://en.wikipedia.org/wiki/Kriging), with support for ordinary
  kriging, kriging with external drift (KED) and block kriging.
- **Radar adjustment** that combines radar fields with gauge and CML
  observations, either additively (interpolating the difference) or
  multiplicatively (interpolating the ratio), using any of the interpolation
  methods above, or directly via KED.
- **Line geometry handling** that treats CMLs correctly: radar values along a
  link are obtained via grid intersection for computing differences, and
  (optional) block kriging uses the full line geometry rather than just the
  midpoint.

## Installation

```bash
pip install mergeplg
```

## Documentation

See the [documentation](https://mergeplg.readthedocs.io/) for usage examples and
the API reference.

## License

Distributed under the terms of the
[BSD-3-Clause](https://opensource.org/license/bsd-3-clause) license.

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/OpenSenseAction/mergeplg/workflows/CI/badge.svg
[actions-link]:             https://github.com/OpenSenseAction/mergeplg/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/mergeplg
[conda-link]:               https://github.com/conda-forge/mergeplg-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/OpenSenseAction/mergeplg/discussions
[pypi-link]:                https://pypi.org/project/mergeplg/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/mergeplg
[pypi-version]:             https://img.shields.io/pypi/v/mergeplg
[rtd-badge]:                https://readthedocs.org/projects/mergeplg/badge/?version=latest
[rtd-link]:                 https://mergeplg.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
