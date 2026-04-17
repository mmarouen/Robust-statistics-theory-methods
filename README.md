# Robust Statistics Theory and Methods

This repository contains Python implementations of tables and figures from a well-known statistics textbook. The goal is to provide reproducible, executable code for visualizations and results presented in the book.

---

## Structure

The repository is organized by chapter:
- chapter_1.py
- chapter_2.py
- chapter_3.py
- ...

Each file corresponds to a chapter and contains functions that generate specific tables or figures.

### Naming Convention

- Functions are named according to the book’s structure:
  - `figure_<chapter>_<number>()`
  - `table_<chapter>_<number>()`

Example:

```python
def figure_3_1():
    ...

def figure_3_2():
    ...
```
---

## Usage

Each chapter file is executable as a standalone script. To run a specific figure or table, you need to manually select the function inside the file.

Example:
```python
if __name__ == '__main__':
    figure_3_1()
```
Run from the command line:

`python chapter_3.py`

You can modify the __main__ block to call any function you want to generate.

---

### Example
```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from rstm import madn, mdn, iqr
from rstm.estimators import Huber, Bisquare, TrimmedMean, Category

def figure_3_1():
    # implementation here
    ...

if __name__ == '__main__':
    figure_3_1()
```
---

## Contributing

Reach out to [mmmarouen](https://github.com/mmarouen)

---

## License

[LICENSE](LICENSE)