<h1 align="center"><strong>DiffCLIP-CR: Image-to-Scalar CLIP Conditioning for Effective Intensity-Guided Cloud Removal in Diffusion Models</strong></h1>
<h3 align="center">Welcome to the official repository of the DiffCLIP-CR. This repository was adapted and modified from DiffCR's <a href="https://github.com/XavierJiezou/DiffCR/blob/main/README.md">repository, which is linked here.</a></h3>
<br>
<h2>Downloading Datasets, Requirements, and Compatibility</h2>
<br>
<p>
Data used in this study was extracted from the Sen2_MTC_New dataset, which can be found with the following link: <a href="https://drive.google.com/file/d/1-hDX9ezWZI2OtiaGbE8RrKJkN1X-ZO1P/view?usp=share_link">Download CTGAN dataset</a>. 
<br><br>
A python version of <strong>3.10</strong> is recommended to run this code. Additionally, CUDA-compatible torch versions are recommended due to the need of manipulating and analyzing large image datasets (namely CTGAN). 
Windows x64 is also recommended for running code in this repository. 
<br><br>
<h3>Requirements: </h3> All requirements for the project can be located under "requirements.txt", which can be installed through the following terminal command:
  <br><br>
<pre><code>pip install -r requirements.txt</code></pre>

Note: Pip automatically installs a torch version without CUDA; please visit the <a href="https://developer.nvidia.com/cuda/toolkit">CUDA website</a> at NVIDIA to download CUDA before installing project requirements. <br><br>
</p>
<h2>Running Training and Testing Commands: </h2>
<br>
<p>
Upon downloading datasets, make sure that all paths to the dataset reference the data folder. If done correctly, this should contain train, val, and test txt files as well as a "Sen2_MTC" folder containing dataset images. 
<br><br>
To run training, run the following command in terminal. Change any training settings under their respective categories in the config/ours_sigmoid.json file. 
<br>
<pre><code>python run.py -p train -c config/ours_sigmoid.json</code></pre>
<br>
Testing can be run with the following command. To ensure the functionality of testing, please set the proper resume state in the config/ours_sigmoid.json file. 
<br><br>
<pre><code>python run.py -p test -c config/ours_sigmoid.json</code></pre>
<br>
To download the pretrained 1000-epoch batch size 2 results for DiffCLIP-CR, please visit <a href="https://drive.google.com/file/d/1x0l02NBBgB25vPVxPP71j1JeSXep89rl/view">here</a>.
</p>

<h2>Acknowledgements</h2>
<br>
<p><a href="https://github.com/XavierJiezou/DiffCR/blob/main/README.md">https://github.com/XavierJiezou/DiffCR</a>
<br><br>
<a href="https://github.com/openai/CLIP">https://github.com/openai/CLIP</a></p>
