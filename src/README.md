
# /src/


### Directories

- **examples/** directory for ready-made examples
- **legacy/** old code kept for legacy reasons
- **thirdparties/** source code developed by third parties

### Basic classes
These files contain the basic code to work with mappings between SCMs. Most of them are implemented only for Rischel abstractions.

- **SCMMappings_1_1.py:** classes for mappings between SCMs (Rischel abstraction, Rubenstein transformation)
- **printing.py:** classes for printing and plotting mappings between SCMs
- **evaluating.py:** classes to evaluate mappings between SCMs (errors, information loss, EI)
- **learning.py:** classes for basic learning of mappings between SCMs (enumeration)

- **MechMappings.py** class for mappings between mechanisms (Hoel emergence) (TO BE REVIEWED)

### Support classes
These files provides additional functionalities.

- **evaluationsets.py:** methods to generates disjoint set of variables X,Y for interventions P(Y|do(X))
- **utils.py:** generic useful functions

### Neural network classes
These files contain code to learn Rischel abstractions using NN.

- **nn.py:** class to implement a JointNN learning an abstraction
- **nn_layers.py:** custom layers for our NN (binarylinear)
- **nn_losses.py:** custom losses for our NN (JSD, rowmax_penalty)

### Genetic algorithms classes
These files contain code to learn Rischel abstractions using GA.

- **ga.py:** class to implement a GA learning an abstraction
- **ga_fitnesses.py:** custom fitness for our NN (JSD, addittive_surjective_penalty)