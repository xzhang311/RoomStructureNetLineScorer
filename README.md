# RoomStructureNetLineScorer

## Overview
RoomStructureNetLineScorer is a Python implementation of the paper, *["RoomStructNet: Learning to Rank Non-Cuboidal Room Layouts From Single View"](https://arxiv.org/abs/2110.00644),* published in 2021. This project focuses on a deep learning model that ranks non-cuboidal room layouts from a single-view image, enabling accurate estimation of complex indoor structures. It is designed for applications in computer vision, 3D scene understanding, and architectural analysis.

## Features
- **Room Layout Ranking**: Learns to rank non-cuboidal room layouts based on single-view images.
- **Line-Based Scoring**: Utilizes line segments to evaluate and score room structure hypotheses.
- **Deep Learning Architecture**: Employs a neural network to process visual and geometric cues.
- **Visualization Support**: Includes tools to visualize predicted layouts and scoring results.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/xzhang311/RoomStructureNetLineScorer.git
   ```
2. Navigate to the project directory:
   ```bash
   cd RoomStructureNetLineScorer
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script to process an input image and rank room layouts:
```bash
python main.py --input path/to/image.jpg --output path/to/results
```
Use the `--help` flag for detailed configuration options:
```bash
python main.py --help
```

## Requirements
- Python 3.8+
- Libraries: PyTorch, NumPy, OpenCV, Matplotlib (listed in `requirements.txt`)

## Reference
This project implements the methodology described in:
- *["RoomStructNet: Learning to Rank Non-Cuboidal Room Layouts From Single View"](https://arxiv.org/abs/2110.00644),* CoRR, 2021. [arXiv:2110.00644](https://arxiv.org/abs/2110.00644)

### Citation
If you find this work useful, please cite it using the following BibTeX:
```bibtex
@article{DBLP:journals/corr/abs-2110-00644,
  author       = {Xi Zhang and
                  Chun{-}Kai Wang and
                  Kenan Deng and
                  Tomas F. Yago Vicente and
                  Himanshu Arora},
  title        = {RoomStructNet: Learning to Rank Non-Cuboidal Room Layouts From Single
                  View},
  journal      = {CoRR},
  volume       = {abs/2110.00644},
  year         = {2021},
  url          = {https://arxiv.org/abs/2110.00644},
  eprinttype   = {arXiv},
  eprint       = {2110.00644},
  timestamp    = {Tue, 27 Feb 2024 16:41:39 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2110-00644.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with your enhancements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
This work is based on the research presented in the 2021 paper and leverages open-source computer vision and deep learning frameworks.
