class DynamicMatrix:
    def __init__(self, rows, cols, default_value=0):
        if rows <= 0 or cols <= 0:
            raise ValueError("Matrix dimensions must be greater than zero.")
        self.matrix = [[default_value for _ in range(cols)] for _ in range(rows)]

    @classmethod
    def from_user_input(cls):
        rows = cls.get_valid_int("Enter number of rows: ")
        cols = cls.get_valid_int("Enter number of columns: ")
        matrix = cls(rows, cols)
        matrix.set_matrix(cls.get_matrix_input(rows, cols))
        return matrix

    @staticmethod
    def get_valid_int(prompt):
        while True:
            try:
                return int(input(prompt))
            except ValueError:
                print("Invalid input! Please enter an integer.")

    @staticmethod
    def get_matrix_input(rows, cols):
        matrix = []
        print(f"Enter the {rows}x{cols} matrix values row by row:")
        for i in range(rows):
            while True:
                try:
                    row = list(map(int, input(f"Row {i + 1}: ").split()))
                    if len(row) != cols:
                        raise ValueError(f"You must enter exactly {cols} values.")
                    break
                except ValueError:
                    print("Invalid input! Please enter integers only.")
            matrix.append(row)
        return matrix

    def display(self):
        for row in self.matrix:
            print(row)
        print()

    def validate_index(self, index, max_value, index_type, for_insert=False):
        if for_insert:
            if index < 0 or index > max_value:
                print(f"Invalid {index_type} index! Valid indices are between 0 and {max_value}.")
                return False
        else:
            if index < 0 or index >= max_value:
                if max_value == 1:
                    print(f"Invalid {index_type} index! There is only one {index_type} with index 0.")
                else:
                    print(f"Invalid {index_type} index! Valid indices are between 0 and {max_value}.")
                return False
        return True

    def handle_insert_row(self):
        index = self.get_valid_int("Enter row index to insert: ")
        if not self.validate_index(index, len(self.matrix), "row", for_insert=True):
            return
        if not self.matrix:
            cols = self.get_valid_int("Enter number of columns for the new row: ")
            new_row = self.get_matrix_input(1, cols)[0]
            self.matrix.append(new_row)
        else:
            new_row = self.get_matrix_input(1, len(self.matrix[0]))[0]
            self.matrix.insert(index, new_row)

    def handle_insert_column(self):
        index = self.get_valid_int("Enter column index to insert: ")
        if not self.validate_index(index, len(self.matrix[0]) if self.matrix else 0, "column", for_insert=True):
            return
        if not self.matrix:
            rows = self.get_valid_int("Enter number of rows for the new column: ")
            new_col = [self.get_valid_int(f"Enter value for row {i + 1}: ") for i in range(rows)]
            self.matrix = [[new_col[i]] for i in range(rows)]
        else:
            new_col = [self.get_valid_int(f"Enter value for row {i + 1}: ") for i in range(len(self.matrix))]
            for i in range(len(self.matrix)):
                self.matrix[i].insert(index, new_col[i])

    def handle_delete_row(self):
        if not self.matrix:
            print("Matrix is empty.")
            return
        index = self.get_valid_int("Enter row index to delete: ")
        if not self.validate_index(index, len(self.matrix), "row"):
            return
        del self.matrix[index]
        if not self.matrix:
            self.matrix = []

    def handle_delete_column(self):
        if not self.matrix:
            print("Matrix is empty.")
            return
        index = self.get_valid_int("Enter column index to delete: ")
        if not self.validate_index(index, len(self.matrix[0]), "column"):
            return
        for row in self.matrix:
            del row[index]
        if not self.matrix[0]:
            self.matrix = []

    def handle_update_element(self):
        if not self.matrix:
            print("Matrix is empty.")
            return
        row = self.get_valid_int("Enter row index: ")
        col = self.get_valid_int("Enter column index: ")
        if not self.validate_index(row, len(self.matrix), "row") or not self.validate_index(col, len(self.matrix[0]), "column"):
            return
        new_value = self.get_valid_int("Enter new value: ")
        self.matrix[row][col] = new_value

    def handle_multiply(self):
        rows2 = self.get_valid_int("Enter number of rows for the second matrix: ")
        cols2 = self.get_valid_int("Enter number of columns for the second matrix: ")
        if len(self.matrix[0]) != rows2:
            print("Matrix multiplication not possible! Number of columns in the first matrix must equal the number of rows in the second matrix.")
            return
        print("Enter second matrix:")
        matrix2 = DynamicMatrix(rows2, cols2)
        matrix2.set_matrix(DynamicMatrix.get_matrix_input(rows2, cols2))
        result = [[sum(self.matrix[i][k] * matrix2.matrix[k][j] for k in range(len(self.matrix[0])))
                for j in range(len(matrix2.matrix[0]))] for i in range(len(self.matrix))]
        self.matrix = result
        self.display()

    def transpose(self):
        if not self.matrix or not self.matrix[0]:
            print("Matrix is empty.")
            return self
        transposed = [[self.matrix[j][i] for j in range(len(self.matrix))] for i in range(len(self.matrix[0]))]
        self.matrix = transposed
        return self

    def set_matrix(self, new_matrix):
        if not all(len(row) == len(new_matrix[0]) for row in new_matrix):
            raise ValueError("All rows in the new matrix must have the same length.")
        self.matrix = new_matrix
        return self