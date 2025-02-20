# Dynamic Matrix Project

This project implements a dynamic matrix class in Python, allowing for the manipulation of matrices through various operations. The main features include:

- Inserting and deleting rows and columns dynamically.
- Multiplying two matrices.
- Computing the transpose of a matrix.
- Efficient handling of edge cases, such as operations on empty matrices or invalid dimensions.

## Installation

To get started with this project, clone the repository and navigate to the project directory. You can install the required dependencies using the following command:

```
pip install -r requirements.txt
```

## Usage

The dynamic matrix functionality is implemented in the `src/matrix.py` file. You can create an instance of the `DynamicMatrix` class and use its methods to manipulate matrices.

### Example

```python
from src.matrix import DynamicMatrix

# Create a new dynamic matrix
matrix = DynamicMatrix()

# Insert a row
matrix.insert_row([1, 2, 3])

# Insert a column
matrix.insert_column([4, 5])

# Multiply two matrices
result = matrix.multiply(other_matrix)

# Transpose the matrix
transposed = matrix.transpose()
```

## Running Tests

To ensure the functionality of the dynamic matrix class, unit tests are provided in the `tests/test_matrix.py` file. You can run the tests using the following command:

```
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.