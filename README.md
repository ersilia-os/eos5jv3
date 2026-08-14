# MycoPermeNet

Predicts the permeation of small molecules across the mycomembrane (outer membrane) of _Mycobacterium tuberculosis_ from a SMILES string. This MycoPermeNet-v2 model fuses a graph neural network embedding with normalized RDKit descriptors and a multilayer perceptron, trained with a Fusion Noisy Student Self-Distillation strategy. Lower scores indicate higher permeability. Applicability domain -- trained on small azide-tagged compounds (MW ~82-570, up to ~50 heavy atoms); predictions for larger, out-of-domain molecules may fall outside the -3.1 to +1.6 output range and be unreliable.

This model was incorporated on 2026-07-09.Last packaged on 2026-08-12.

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
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos5jv3](https://hub.docker.com/r/ersiliaos/eos5jv3)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos5jv3.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos5jv3.zip)

### Resource Consumption
- **Model Size (Mb):** `2`
- **Environment Size (Mb):** `1674`
- **Image Size (Mb):** `1658.99`

**Computational Performance (seconds):**
- 10 inputs: `38.85`
- 100 inputs: `23.21`
- 10000 inputs: `148.64`

### References
- **Source Code**: [https://github.com/SAGE-Lab-UMass/MycoPermeNet-v2-pub](https://github.com/SAGE-Lab-UMass/MycoPermeNet-v2-pub)
- **Publication**: [https://doi.org/10.1021/acs.jcim.5c02435](https://doi.org/10.1021/acs.jcim.5c02435)
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
