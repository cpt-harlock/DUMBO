This repository contains the training, testing and simulator code used to build the DUMBO system from the research paper [Taming the Elephants: Affordable Flow Length Prediction in the Data Plane](https://dl.acm.org/doi/10.1145/3649473).

![dumbo_intro_fig](https://github.com/cpt-harlock/DUMBO/blob/main/dumbo_intro.png)

DUMBO is a versatile networked system that integrates a lightweight traffic classifier to enhance several downstream tasks in the data plane (e.g., packet scheduling, inter-arrival times distribution estimation, flow length estimation). 
The main idea of DUMBO is to segregate elephants and mice flows to address them separately, hence saving memory and improving performance over standard baselines.

## Introduction
This document serves as a guide to install and use the DUMBO system on real traffic traces.

## Quickstart
Follow these instructions to quickly set up the repository and reproduce the experiments on Linux (Ubuntu version >= 22).

1. **Dependencies**
   - Install [mergecap](https://www.wireshark.org/docs/man-pages/mergecap.html) and [editcap](https://www.wireshark.org/docs/man-pages/editcap.html)
      ```bash
      $ sudo apt-get install wireshark-common
      ```
   - Install Python 3.9 outside of any virtual environment
      ```bash
      $ sudo apt update
      $ sudo apt install python3.9
      $ python --version
      ```
   - Install and setup [Rust](https://www.rust-lang.org/tools/install)

      1. Use ```v1.76.0-nightly``` and check your version:
      ```bash 
      $ cargo --version
      ```
      2. Install the ```libpython3.9-dev``` package on your system:
      ```bash
      $ sudo apt install libpython3.9-dev
      ```
      3. Deactivate any virtual environment and build the repository:
      ```
      $ cargo build -r
      ```

   - Create the required [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) environments
      ```bash
      $ chmod +x ./setup_conda.sh
      $ ./setup_conda.sh
      ```

2. **Data**

Download the traces (see *Traffic traces* below). Uncompress and store the ```*.pcap``` files in the appropriate folder:
   - ```./data/caida/pcap/equinix-chicago.dirA.20160121-{hour}.UTC.anon.pcap```
   - ```./data/mawi/pcap/20190409{hour}.pcap```
   - ```./data/uni/pcap/univ2.pcap```

3. **Scheduling simulator**

Clone and patch the YAPS simulator repository
   ```bash
   $ git clone -n https://github.com/NetSys/simulator.git
   $ cd simulator
   $ git checkout -b scheduling_DUMBO 179b64e
   $ git apply < ../scheduling_DUMBO.patch
   $ cd ..
   ```

5. **Run**

Run the pipeline to reproduce the experiments:
   ```bash
   $ chmod +x ./run.sh
   $ ./run.sh caida  # Includes trade-off analysis
   $ ./run.sh mawi
   $ ./run.sh uni
   $ chmod +x ./run_update_stresstest.sh
   $ ./run_update_stresstest.sh # Requires complete caida and mawi runs
   ```
   
6. **Plot**

Plot the results using the notebooks in ```./plots/```

## Traffic traces
Here are the data used in the experiments.
#### CAIDA
- Trace: equinix Chicago dir.A 2016-01-21 13:00 - 13:59
- Link: https://www.caida.org/catalog/datasets/passive_dataset_download/ (approval required by CAIDA)

#### MAWI
- Trace: 2019-04-09 18:30 - 19:45
- Link: https://mawi.wide.ad.jp/mawi/ditl/ditl2019/

#### UNI
- Trace: UNI2 2010-01-22 20:02 - 22:40
- Link: https://pages.cs.wisc.edu/~tbenson/IMC_DATA/univ2_trace.tgz

## Documentation
You can find additional technical documentation about the simulators in ```./README_SIMULATOR.md``` and ```./README_DEV.md```.

## Citation
If you have found this paper useful, please cite us using: 
```
@article{dumbo2024,
  title={Taming the Elephants: Affordable Flow Length Prediction in the Data Plane},
  author={Azorin, Raphael and Monterubbiano, Andrea and Castellano, Gabriele and Gallo, Massimo and Pontarelli, Salvatore and Rossi, Dario},
  journal={Proceedings of the ACM on Networking},
  volume={2},
  number={CoNEXT1},
  articleno = {5},
  numpages={24},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```

## Ackowledgements

We would like to thank the authors of [pHost](https://dl.acm.org/doi/10.1145/2716281.2836086) and of the [YAPS](https://github.com/NetSys/simulator) simulator as well as the author of the [MetaCost learning implementation](https://github.com/Treers/MetaCost/tree/master).
