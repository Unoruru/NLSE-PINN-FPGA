## 20260315_complex_v3 Bitfiles

This folder contains the files required for FINN to regenerate the bitfiles. Please see the releases [here](https://github.com/Anthony062966/PINN_FPGA_Project/releases/tag/complex_pynqzu_v3) for the appropriate pre-generated bitfiles.

See ``../20260315_complex_v2/_scripts`` for the scripts required to execute inputs on the accelerator. Appropriate generated input signal files are required to run the scripts.

Evaluation script (temp) for checking output results available at `evalComplexTemp.py`. See the following section.

The corresponding release is titled: `v3-20260322: complex bitfiles for all supported signal type targetting PYNQ-ZU`.

## Evaluation Script (Temp)
The script generates the constellation diagram and metrics (EVM & SER) for the FPGA accelerator generated outputs. To start, run from the project root:
```bash
python pynq-zu/_deployment/20260322_complex_v3/evalComplexTemp.py --sig_type {sig_type}
```
Replace {sig_type} with one of: [`16apsk`, `16psk`, `16qam`, `star`]. By default, `16qam` is ran.

See `/results` for the generated outputs.
