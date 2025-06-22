This repository is intended to replicate the results and generate the stimuli used in our visual search experiments. The codebase is still under maintenance and may require some cleanup, but all major scripts are available for review and use.

## Repository Structure

- generate_stimuli  
  Contains the code used to generate the visual search arrays employed in the experiments.

- generate_template  
  Code to generate fixed and generative templates.  
  For generative templates, please:
  1. Download this [repo](https://github.com/ggaziv/Wormholes) and their model checkpoints.
  2. Place the gen_v10.py file under the perturb folder.
  3. Run `python -m scripts.generate_target_modulated_images --num 0 --batch_size 100 --gen_version 10`

- eval  
  Scripts to evaluate:
  - Search accuracy under various conditions
  - Comparison with human search behavior, including guidance metrics, spatial fixation maps, and shape-based attentional guidance.

- preprocess_gazedata  
  Code to preprocess fixation data collected from the search tasks. Preprocessed data are stored in the `data` directory.


## Citation
TBD
