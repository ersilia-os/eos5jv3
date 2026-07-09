# MycoPermeNet

Predicts the permeation of small molecules across the mycomembrane (the mycobacterial outer membrane) of _Mycobacterium tuberculosis_. Taking a SMILES string as input, this deep-learning model (a multilayer perceptron over learned chemical embeddings) outputs a standardized-residual permeability score. Higher permeation is associated with nitrogen-containing aromatic scaffolds such as indole. Useful for prioritizing anti-TB compounds likely to accumulate within Mtb cells.



## Information
### Identifiers
- **Ersilia Identifier:** `eos5jv3`
- **Slug:** `mycopermenet`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Activity prediction`
- **Biomedical Area:** `Tuberculosis`
- **Target Organism:** `Mycobacterium tuberculosis`
- **Tags:** `Permeability`, `Antimicrobial activity`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `1`
- **Output Consistency:** `Fixed`
- **Interpretation:** Predicted standardized residual of mycomembrane permeation; lower values indicate higher permeability in M. tuberculosis.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| mycomembrane_permeation | float | low | Predicted standardized residual of mycomembrane permeation in Mycobacterium tuberculosis where lower values indicate higher permeability |


### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`

### Resource Consumption


### References
- **Source Code**: [https://github.com/Nevbarunegbe/Mycomembrane-permeability-project](https://github.com/Nevbarunegbe/Mycomembrane-permeability-project)
- **Publication**: [https://doi.org/10.1038/s41564-026-02412-5](https://doi.org/10.1038/s41564-026-02412-5)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2026`
- **Ersilia Contributor:** [GemmaTuron](https://github.com/GemmaTuron)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [MIT](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos5jv3
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos5jv3
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
