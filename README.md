## Project Aim

The aim of this project is to implement Mutation testing on our own implementation of software which facilitates calculations in various numerical bases and various matrix operations.

## Tools Used

MutPy: For mutation testing.
VsCode: Ide for development
Pytest: For executing test cases.
Unittest: For making test cases.

## Mutation Operators Used at Unit level

- AOR
- ASR
- COR
- COI
- LOR
- ROR

## Mutation Operators Used at Integration level

- Integration Parameter Exchange (IPEX)
- Integration Method Call Deletion (IMCD)
- Integration Return Expression Modification (IREM)

## Command to execute mutation testing
mut.py --target source_code --unit-test test -m <br />
Here: <br />
    - source_code: Name of the python file containing source code
    - test: Name of the python file containing tests


## Contribution

- Arya Kondawar:
    Implemented Methods of large multiplication, large division, real division, vector norm, SVD, reduced SVD.
- Naitik Solanki:
    Implemented Methods of large addition, large subtraction,, rank of a matrix, eigen values and vectors of a matrix.
