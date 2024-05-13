.. code:: python

    #!pip list
    
    # this is the liat of dependencies


.. parsed-literal::

    Package                 Version
    ----------------------- ----------
    archspec                0.2.2
    boltons                 23.1.1
    Brotli                  1.1.0
    certifi                 2023.11.17
    cffi                    1.16.0
    charset-normalizer      3.3.2
    colorama                0.4.6
    conda                   23.11.0
    conda-libmamba-solver   23.12.0
    conda-package-handling  2.2.0
    conda_package_streaming 0.9.0
    distro                  1.8.0
    idna                    3.6
    jsonpatch               1.33
    jsonpointer             2.4
    libmambapy              1.5.5
    mamba                   1.5.5
    menuinst                2.0.1
    packaging               23.2
    pip                     23.3.2
    platformdirs            4.1.0
    pluggy                  1.3.0
    pycosat                 0.6.6
    pycparser               2.21
    PySocks                 1.7.1
    requests                2.31.0
    ruamel.yaml             0.18.5
    ruamel.yaml.clib        0.2.7
    setuptools              68.2.2
    tqdm                    4.66.1
    truststore              0.8.0
    urllib3                 2.1.0
    wheel                   0.42.0
    zstandard               0.22.0
    


import
------

.. code:: python

    import multiprocessing
    multiprocessing.cpu_count()




.. parsed-literal::

    2



.. code:: python

    !pip install -i https://test.pypi.org/simple/ ChemicalDice==0.2.8


.. parsed-literal::

    Looking in indexes: https://test.pypi.org/simple/
    Collecting ChemicalDice==0.2.8
      Downloading https://test-files.pythonhosted.org/packages/6b/10/af075721f6e17743e228004bfa97c5b84b8b57dd1b30ee75d0d08fc75870/ChemicalDice-0.2.8-py3-none-any.whl (151 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m151.5/151.5 kB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: ChemicalDice
      Attempting uninstall: ChemicalDice
        Found existing installation: ChemicalDice 0.2.7
        Uninstalling ChemicalDice-0.2.7:
          Successfully uninstalled ChemicalDice-0.2.7
    Successfully installed ChemicalDice-0.2.8
    

.. code:: python

    !pip install signaturizer


.. parsed-literal::

    Requirement already satisfied: signaturizer in /usr/local/lib/python3.10/dist-packages (1.1.14)
    Requirement already satisfied: tensorflow in /usr/local/lib/python3.10/dist-packages (from signaturizer) (2.15.0)
    Requirement already satisfied: tensorflow-hub in /usr/local/lib/python3.10/dist-packages (from signaturizer) (0.16.1)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from signaturizer) (4.66.4)
    Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (1.4.0)
    Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (1.6.3)
    Requirement already satisfied: flatbuffers>=23.5.26 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (24.3.25)
    Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (0.5.4)
    Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (0.2.0)
    Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (3.9.0)
    Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (18.1.1)
    Requirement already satisfied: ml-dtypes~=0.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (0.2.0)
    Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (1.25.2)
    Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (3.3.0)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (24.0)
    Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (3.20.3)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (67.7.2)
    Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (1.16.0)
    Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (2.4.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (4.11.0)
    Requirement already satisfied: wrapt<1.15,>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (1.14.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (0.37.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (1.63.0)
    Requirement already satisfied: tensorboard<2.16,>=2.15 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (2.15.2)
    Requirement already satisfied: tensorflow-estimator<2.16,>=2.15.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (2.15.0)
    Requirement already satisfied: keras<2.16,>=2.15.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->signaturizer) (2.15.0)
    Requirement already satisfied: tf-keras>=2.14.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow-hub->signaturizer) (2.15.1)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow->signaturizer) (0.43.0)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow->signaturizer) (2.27.0)
    Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow->signaturizer) (1.2.0)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow->signaturizer) (3.6)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow->signaturizer) (2.31.0)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow->signaturizer) (0.7.2)
    Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow->signaturizer) (3.0.3)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow->signaturizer) (5.3.3)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow->signaturizer) (0.4.0)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow->signaturizer) (4.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow->signaturizer) (1.3.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow->signaturizer) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow->signaturizer) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow->signaturizer) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow->signaturizer) (2024.2.2)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow->signaturizer) (2.1.5)
    Requirement already satisfied: pyasn1<0.7.0,>=0.4.6 in /usr/local/lib/python3.10/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow->signaturizer) (0.6.0)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow->signaturizer) (3.2.2)
    

.. code:: python

    !pip install rdkit


.. parsed-literal::

    Requirement already satisfied: rdkit in /usr/local/lib/python3.10/dist-packages (2023.9.6)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from rdkit) (1.25.2)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.10/dist-packages (from rdkit) (9.4.0)
    

.. code:: python

    !pip install descriptastorus


.. parsed-literal::

    Collecting descriptastorus
      Downloading descriptastorus-2.6.1-py3-none-any.whl (1.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m10.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pandas-flavor (from descriptastorus)
      Downloading pandas_flavor-0.6.0-py3-none-any.whl (7.2 kB)
    Requirement already satisfied: rdkit in /usr/local/lib/python3.10/dist-packages (from descriptastorus) (2023.9.6)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from descriptastorus) (1.11.4)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from descriptastorus) (1.25.2)
    Requirement already satisfied: pandas>=0.23 in /usr/local/lib/python3.10/dist-packages (from pandas-flavor->descriptastorus) (2.0.3)
    Requirement already satisfied: xarray in /usr/local/lib/python3.10/dist-packages (from pandas-flavor->descriptastorus) (2023.7.0)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.10/dist-packages (from rdkit->descriptastorus) (9.4.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.23->pandas-flavor->descriptastorus) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.23->pandas-flavor->descriptastorus) (2023.4)
    Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.23->pandas-flavor->descriptastorus) (2024.1)
    Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.10/dist-packages (from xarray->pandas-flavor->descriptastorus) (24.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas>=0.23->pandas-flavor->descriptastorus) (1.16.0)
    Installing collected packages: pandas-flavor, descriptastorus
    Successfully installed descriptastorus-2.6.1 pandas-flavor-0.6.0
    

.. code:: python

    !pip install mordred


.. parsed-literal::

    Collecting mordred
      Downloading mordred-1.2.0.tar.gz (128 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m128.8/128.8 kB[0m [31m3.3 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
    Requirement already satisfied: six==1.* in /usr/local/lib/python3.10/dist-packages (from mordred) (1.16.0)
    Requirement already satisfied: numpy==1.* in /usr/local/lib/python3.10/dist-packages (from mordred) (1.25.2)
    Collecting networkx==2.* (from mordred)
      Downloading networkx-2.8.8-py3-none-any.whl (2.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m14.4 MB/s[0m eta [36m0:00:00[0m
    [?25hBuilding wheels for collected packages: mordred
      Building wheel for mordred (setup.py) ... [?25l[?25hdone
      Created wheel for mordred: filename=mordred-1.2.0-py3-none-any.whl size=176720 sha256=3f95b4ce6c114705e27a79801d59e759eed32905ae7420ae3fd7420bc83f70ee
      Stored in directory: /root/.cache/pip/wheels/a7/4f/b8/d4c6591f6ac944aaced7865b349477695f662388ad958743c7
    Successfully built mordred
    Installing collected packages: networkx, mordred
      Attempting uninstall: networkx
        Found existing installation: networkx 3.3
        Uninstalling networkx-3.3:
          Successfully uninstalled networkx-3.3
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    torch 2.2.1+cu121 requires nvidia-cublas-cu12==12.1.3.1; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
    torch 2.2.1+cu121 requires nvidia-cuda-cupti-cu12==12.1.105; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
    torch 2.2.1+cu121 requires nvidia-cuda-nvrtc-cu12==12.1.105; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
    torch 2.2.1+cu121 requires nvidia-cuda-runtime-cu12==12.1.105; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
    torch 2.2.1+cu121 requires nvidia-cudnn-cu12==8.9.2.26; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
    torch 2.2.1+cu121 requires nvidia-cufft-cu12==11.0.2.54; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
    torch 2.2.1+cu121 requires nvidia-curand-cu12==10.3.2.106; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
    torch 2.2.1+cu121 requires nvidia-cusolver-cu12==11.4.5.107; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
    torch 2.2.1+cu121 requires nvidia-cusparse-cu12==12.1.0.106; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
    torch 2.2.1+cu121 requires nvidia-nccl-cu12==2.19.3; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
    torch 2.2.1+cu121 requires nvidia-nvtx-cu12==12.1.105; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.[0m[31m
    [0mSuccessfully installed mordred-1.2.0 networkx-2.8.8
    

.. code:: python

    !pip install tensorly


.. parsed-literal::

    Collecting tensorly
      Downloading tensorly-0.8.1-py3-none-any.whl (229 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m229.7/229.7 kB[0m [31m5.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from tensorly) (1.25.2)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from tensorly) (1.11.4)
    Installing collected packages: tensorly
    Successfully installed tensorly-0.8.1
    

.. code:: python

    !pip install -q condacolab
    import condacolab
    condacolab.install()


.. parsed-literal::

    ⏬ Downloading https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Mambaforge-23.11.0-0-Linux-x86_64.sh...
    📦 Installing...
    📌 Adjusting configuration...
    🩹 Patching environment...
    ⏲ Done in 0:00:17
    🔁 Restarting kernel...
    

.. code:: python

    !pip install rdkit
    !pip install signaturizer
    !pip install descriptastorus
    !pip install mordred
    !pip install tensorly


.. parsed-literal::

    Collecting rdkit
      Downloading rdkit-2023.9.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.9 kB)
    Collecting numpy (from rdkit)
      Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m61.0/61.0 kB[0m [31m3.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting Pillow (from rdkit)
      Downloading pillow-10.3.0-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (9.2 kB)
    Downloading rdkit-2023.9.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (34.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m34.9/34.9 MB[0m [31m54.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m18.2/18.2 MB[0m [31m92.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pillow-10.3.0-cp310-cp310-manylinux_2_28_x86_64.whl (4.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.5/4.5 MB[0m [31m101.3 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: Pillow, numpy, rdkit
    Successfully installed Pillow-10.3.0 numpy-1.26.4 rdkit-2023.9.6
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mCollecting signaturizer
      Using cached signaturizer-1.1.14-py3-none-any.whl
    Collecting tensorflow (from signaturizer)
      Downloading tensorflow-2.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.3 kB)
    Collecting tensorflow-hub (from signaturizer)
      Downloading tensorflow_hub-0.16.1-py2.py3-none-any.whl.metadata (1.3 kB)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/site-packages (from signaturizer) (4.66.1)
    Collecting absl-py>=1.0.0 (from tensorflow->signaturizer)
      Downloading absl_py-2.1.0-py3-none-any.whl.metadata (2.3 kB)
    Collecting astunparse>=1.6.0 (from tensorflow->signaturizer)
      Downloading astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
    Collecting flatbuffers>=23.5.26 (from tensorflow->signaturizer)
      Downloading flatbuffers-24.3.25-py2.py3-none-any.whl.metadata (850 bytes)
    Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 (from tensorflow->signaturizer)
      Downloading gast-0.5.4-py3-none-any.whl.metadata (1.3 kB)
    Collecting google-pasta>=0.1.1 (from tensorflow->signaturizer)
      Downloading google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
    Collecting h5py>=3.10.0 (from tensorflow->signaturizer)
      Downloading h5py-3.11.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.5 kB)
    Collecting libclang>=13.0.0 (from tensorflow->signaturizer)
      Downloading libclang-18.1.1-py2.py3-none-manylinux2010_x86_64.whl.metadata (5.2 kB)
    Collecting ml-dtypes~=0.3.1 (from tensorflow->signaturizer)
      Downloading ml_dtypes-0.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
    Collecting opt-einsum>=2.3.2 (from tensorflow->signaturizer)
      Downloading opt_einsum-3.3.0-py3-none-any.whl.metadata (6.5 kB)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/site-packages (from tensorflow->signaturizer) (23.2)
    Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 (from tensorflow->signaturizer)
      Downloading protobuf-4.25.3-cp37-abi3-manylinux2014_x86_64.whl.metadata (541 bytes)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/site-packages (from tensorflow->signaturizer) (2.31.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/site-packages (from tensorflow->signaturizer) (68.2.2)
    Collecting six>=1.12.0 (from tensorflow->signaturizer)
      Downloading six-1.16.0-py2.py3-none-any.whl.metadata (1.8 kB)
    Collecting termcolor>=1.1.0 (from tensorflow->signaturizer)
      Downloading termcolor-2.4.0-py3-none-any.whl.metadata (6.1 kB)
    Collecting typing-extensions>=3.6.6 (from tensorflow->signaturizer)
      Downloading typing_extensions-4.11.0-py3-none-any.whl.metadata (3.0 kB)
    Collecting wrapt>=1.11.0 (from tensorflow->signaturizer)
      Downloading wrapt-1.16.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)
    Collecting grpcio<2.0,>=1.24.3 (from tensorflow->signaturizer)
      Downloading grpcio-1.63.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.2 kB)
    Collecting tensorboard<2.17,>=2.16 (from tensorflow->signaturizer)
      Downloading tensorboard-2.16.2-py3-none-any.whl.metadata (1.6 kB)
    Collecting keras>=3.0.0 (from tensorflow->signaturizer)
      Downloading keras-3.3.3-py3-none-any.whl.metadata (5.7 kB)
    Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow->signaturizer)
      Downloading tensorflow_io_gcs_filesystem-0.37.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (14 kB)
    Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /usr/local/lib/python3.10/site-packages (from tensorflow->signaturizer) (1.26.4)
    Collecting tf-keras>=2.14.1 (from tensorflow-hub->signaturizer)
      Downloading tf_keras-2.16.0-py3-none-any.whl.metadata (1.6 kB)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow->signaturizer) (0.42.0)
    Collecting rich (from keras>=3.0.0->tensorflow->signaturizer)
      Downloading rich-13.7.1-py3-none-any.whl.metadata (18 kB)
    Collecting namex (from keras>=3.0.0->tensorflow->signaturizer)
      Downloading namex-0.0.8-py3-none-any.whl.metadata (246 bytes)
    Collecting optree (from keras>=3.0.0->tensorflow->signaturizer)
      Downloading optree-0.11.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (45 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m45.4/45.4 kB[0m [31m3.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow->signaturizer) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow->signaturizer) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow->signaturizer) (2.1.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow->signaturizer) (2023.11.17)
    Collecting markdown>=2.6.8 (from tensorboard<2.17,>=2.16->tensorflow->signaturizer)
      Downloading Markdown-3.6-py3-none-any.whl.metadata (7.0 kB)
    Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard<2.17,>=2.16->tensorflow->signaturizer)
      Downloading tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl.metadata (1.1 kB)
    Collecting werkzeug>=1.0.1 (from tensorboard<2.17,>=2.16->tensorflow->signaturizer)
      Downloading werkzeug-3.0.3-py3-none-any.whl.metadata (3.7 kB)
    Collecting MarkupSafe>=2.1.1 (from werkzeug>=1.0.1->tensorboard<2.17,>=2.16->tensorflow->signaturizer)
      Downloading MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)
    Collecting markdown-it-py>=2.2.0 (from rich->keras>=3.0.0->tensorflow->signaturizer)
      Downloading markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)
    Collecting pygments<3.0.0,>=2.13.0 (from rich->keras>=3.0.0->tensorflow->signaturizer)
      Downloading pygments-2.18.0-py3-none-any.whl.metadata (2.5 kB)
    Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich->keras>=3.0.0->tensorflow->signaturizer)
      Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
    Downloading tensorflow-2.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (589.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m589.8/589.8 MB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tensorflow_hub-0.16.1-py2.py3-none-any.whl (30 kB)
    Downloading absl_py-2.1.0-py3-none-any.whl (133 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m133.7/133.7 kB[0m [31m11.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Downloading flatbuffers-24.3.25-py2.py3-none-any.whl (26 kB)
    Downloading gast-0.5.4-py3-none-any.whl (19 kB)
    Downloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.5/57.5 kB[0m [31m5.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading grpcio-1.63.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.6/5.6 MB[0m [31m103.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading h5py-3.11.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.3/5.3 MB[0m [31m113.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading keras-3.3.3-py3-none-any.whl (1.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m54.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading libclang-18.1.1-py2.py3-none-manylinux2010_x86_64.whl (24.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.5/24.5 MB[0m [31m81.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading ml_dtypes-0.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.2/2.2 MB[0m [31m83.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m65.5/65.5 kB[0m [31m5.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading protobuf-4.25.3-cp37-abi3-manylinux2014_x86_64.whl (294 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m294.6/294.6 kB[0m [31m26.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading six-1.16.0-py2.py3-none-any.whl (11 kB)
    Downloading tensorboard-2.16.2-py3-none-any.whl (5.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.5/5.5 MB[0m [31m107.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tensorflow_io_gcs_filesystem-0.37.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.1/5.1 MB[0m [31m95.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading termcolor-2.4.0-py3-none-any.whl (7.7 kB)
    Downloading tf_keras-2.16.0-py3-none-any.whl (1.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m79.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading typing_extensions-4.11.0-py3-none-any.whl (34 kB)
    Downloading wrapt-1.16.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (80 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m80.3/80.3 kB[0m [31m7.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading Markdown-3.6-py3-none-any.whl (105 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m105.4/105.4 kB[0m [31m10.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl (6.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.6/6.6 MB[0m [31m110.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading werkzeug-3.0.3-py3-none-any.whl (227 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m227.3/227.3 kB[0m [31m20.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading namex-0.0.8-py3-none-any.whl (5.8 kB)
    Downloading optree-0.11.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (311 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m311.2/311.2 kB[0m [31m26.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading rich-13.7.1-py3-none-any.whl (240 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m240.7/240.7 kB[0m [31m19.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading markdown_it_py-3.0.0-py3-none-any.whl (87 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m87.5/87.5 kB[0m [31m7.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
    Downloading pygments-2.18.0-py3-none-any.whl (1.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m66.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)
    Installing collected packages: namex, libclang, flatbuffers, wrapt, typing-extensions, termcolor, tensorflow-io-gcs-filesystem, tensorboard-data-server, six, pygments, protobuf, opt-einsum, ml-dtypes, mdurl, MarkupSafe, markdown, h5py, grpcio, gast, absl-py, werkzeug, optree, markdown-it-py, google-pasta, astunparse, tensorboard, rich, keras, tensorflow, tf-keras, tensorflow-hub, signaturizer
    Successfully installed MarkupSafe-2.1.5 absl-py-2.1.0 astunparse-1.6.3 flatbuffers-24.3.25 gast-0.5.4 google-pasta-0.2.0 grpcio-1.63.0 h5py-3.11.0 keras-3.3.3 libclang-18.1.1 markdown-3.6 markdown-it-py-3.0.0 mdurl-0.1.2 ml-dtypes-0.3.2 namex-0.0.8 opt-einsum-3.3.0 optree-0.11.0 protobuf-4.25.3 pygments-2.18.0 rich-13.7.1 signaturizer-1.1.14 six-1.16.0 tensorboard-2.16.2 tensorboard-data-server-0.7.2 tensorflow-2.16.1 tensorflow-hub-0.16.1 tensorflow-io-gcs-filesystem-0.37.0 termcolor-2.4.0 tf-keras-2.16.0 typing-extensions-4.11.0 werkzeug-3.0.3 wrapt-1.16.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m



.. parsed-literal::

    Collecting descriptastorus
      Downloading descriptastorus-2.6.1-py3-none-any.whl.metadata (9.9 kB)
    Collecting pandas-flavor (from descriptastorus)
      Downloading pandas_flavor-0.6.0-py3-none-any.whl.metadata (6.3 kB)
    Requirement already satisfied: rdkit in /usr/local/lib/python3.10/site-packages (from descriptastorus) (2023.9.6)
    Collecting scipy (from descriptastorus)
      Downloading scipy-1.13.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (60 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.6/60.6 kB[0m [31m4.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.10/site-packages (from descriptastorus) (1.26.4)
    Collecting pandas>=0.23 (from pandas-flavor->descriptastorus)
      Downloading pandas-2.2.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (19 kB)
    Collecting xarray (from pandas-flavor->descriptastorus)
      Downloading xarray-2024.3.0-py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.10/site-packages (from rdkit->descriptastorus) (10.3.0)
    Collecting python-dateutil>=2.8.2 (from pandas>=0.23->pandas-flavor->descriptastorus)
      Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
    Collecting pytz>=2020.1 (from pandas>=0.23->pandas-flavor->descriptastorus)
      Downloading pytz-2024.1-py2.py3-none-any.whl.metadata (22 kB)
    Collecting tzdata>=2022.7 (from pandas>=0.23->pandas-flavor->descriptastorus)
      Downloading tzdata-2024.1-py2.py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: packaging>=22 in /usr/local/lib/python3.10/site-packages (from xarray->pandas-flavor->descriptastorus) (23.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas>=0.23->pandas-flavor->descriptastorus) (1.16.0)
    Downloading descriptastorus-2.6.1-py3-none-any.whl (1.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m55.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pandas_flavor-0.6.0-py3-none-any.whl (7.2 kB)
    Downloading scipy-1.13.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (38.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m38.6/38.6 MB[0m [31m14.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pandas-2.2.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.0/13.0 MB[0m [31m99.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading xarray-2024.3.0-py3-none-any.whl (1.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m64.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m229.9/229.9 kB[0m [31m21.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pytz-2024.1-py2.py3-none-any.whl (505 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m505.5/505.5 kB[0m [31m38.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tzdata-2024.1-py2.py3-none-any.whl (345 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m345.4/345.4 kB[0m [31m29.2 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: pytz, tzdata, scipy, python-dateutil, pandas, xarray, pandas-flavor, descriptastorus
    [31mERROR: Operation cancelled by user[0m[31m
    [0mCollecting mordred
      Using cached mordred-1.2.0-py3-none-any.whl
    Requirement already satisfied: six==1.* in /usr/local/lib/python3.10/site-packages (from mordred) (1.16.0)
    Requirement already satisfied: numpy==1.* in /usr/local/lib/python3.10/site-packages (from mordred) (1.26.4)
    Collecting networkx==2.* (from mordred)
      Downloading networkx-2.8.8-py3-none-any.whl.metadata (5.1 kB)
    Downloading networkx-2.8.8-py3-none-any.whl (2.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m44.7 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: networkx, mordred
    

.. code:: python

    !conda install openbabel


.. parsed-literal::

    /bin/bash: line 1: conda: command not found
    

.. code:: python

    !conda install conda-forge::cpulimit


.. parsed-literal::

    /bin/bash: line 1: conda: command not found
    

.. code:: python

    !conda install openbabel
    !conda install conda-forge::cpulimit


.. parsed-literal::

    Channels:
     - conda-forge
    Platform: linux-64
    Collecting package metadata (repodata.json): - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - done
    Solving environment: | / - \ done
    
    
    ==> WARNING: A newer version of conda exists. <==
        current version: 23.11.0
        latest version: 24.4.0
    
    Please update conda by running
    
        $ conda update -n base -c conda-forge conda
    
    
    
    ## Package Plan ##
    
      environment location: /usr/local
    
      added / updated specs:
        - openbabel
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        ca-certificates-2024.2.2   |       hbcca054_0         152 KB  conda-forge
        cairo-1.18.0               |       h3faef2a_0         959 KB  conda-forge
        certifi-2024.2.2           |     pyhd8ed1ab_0         157 KB  conda-forge
        expat-2.6.2                |       h59595ed_0         134 KB  conda-forge
        font-ttf-dejavu-sans-mono-2.37|       hab24e00_0         388 KB  conda-forge
        font-ttf-inconsolata-3.000 |       h77eed37_0          94 KB  conda-forge
        font-ttf-source-code-pro-2.038|       h77eed37_0         684 KB  conda-forge
        font-ttf-ubuntu-0.83       |       h77eed37_2         1.5 MB  conda-forge
        fontconfig-2.14.2          |       h14ed4e7_0         266 KB  conda-forge
        fonts-conda-ecosystem-1    |                0           4 KB  conda-forge
        fonts-conda-forge-1        |                0           4 KB  conda-forge
        freetype-2.12.1            |       h267a509_2         620 KB  conda-forge
        libexpat-2.6.2             |       h59595ed_0          72 KB  conda-forge
        libglib-2.80.2             |       hf974151_0         3.7 MB  conda-forge
        libpng-1.6.43              |       h2797004_0         281 KB  conda-forge
        libxcb-1.15                |       h0b41bf4_0         375 KB  conda-forge
        openbabel-3.1.1            |  py310hbff9852_9         5.0 MB  conda-forge
        openssl-3.3.0              |       hd590300_0         2.8 MB  conda-forge
        pcre2-10.43                |       hcad00b1_0         929 KB  conda-forge
        pixman-0.43.2              |       h59595ed_0         378 KB  conda-forge
        pthread-stubs-0.4          |    h36c2ea0_1001           5 KB  conda-forge
        xorg-kbproto-1.0.7         |    h7f98852_1002          27 KB  conda-forge
        xorg-libice-1.1.1          |       hd590300_0          57 KB  conda-forge
        xorg-libsm-1.2.4           |       h7391055_0          27 KB  conda-forge
        xorg-libx11-1.8.9          |       h8ee46fc_0         809 KB  conda-forge
        xorg-libxau-1.0.11         |       hd590300_0          14 KB  conda-forge
        xorg-libxdmcp-1.1.3        |       h7f98852_0          19 KB  conda-forge
        xorg-libxext-1.3.4         |       h0b41bf4_2          49 KB  conda-forge
        xorg-libxrender-0.9.11     |       hd590300_0          37 KB  conda-forge
        xorg-renderproto-0.11.1    |    h7f98852_1002           9 KB  conda-forge
        xorg-xextproto-7.3.0       |    h0b41bf4_1003          30 KB  conda-forge
        xorg-xproto-7.0.31         |    h7f98852_1007          73 KB  conda-forge
        zlib-1.2.13                |       hd590300_5          91 KB  conda-forge
        ------------------------------------------------------------
                                               Total:        19.6 MB
    
    The following NEW packages will be INSTALLED:
    
      cairo              conda-forge/linux-64::cairo-1.18.0-h3faef2a_0 
      expat              conda-forge/linux-64::expat-2.6.2-h59595ed_0 
      font-ttf-dejavu-s~ conda-forge/noarch::font-ttf-dejavu-sans-mono-2.37-hab24e00_0 
      font-ttf-inconsol~ conda-forge/noarch::font-ttf-inconsolata-3.000-h77eed37_0 
      font-ttf-source-c~ conda-forge/noarch::font-ttf-source-code-pro-2.038-h77eed37_0 
      font-ttf-ubuntu    conda-forge/noarch::font-ttf-ubuntu-0.83-h77eed37_2 
      fontconfig         conda-forge/linux-64::fontconfig-2.14.2-h14ed4e7_0 
      fonts-conda-ecosy~ conda-forge/noarch::fonts-conda-ecosystem-1-0 
      fonts-conda-forge  conda-forge/noarch::fonts-conda-forge-1-0 
      freetype           conda-forge/linux-64::freetype-2.12.1-h267a509_2 
      libexpat           conda-forge/linux-64::libexpat-2.6.2-h59595ed_0 
      libglib            conda-forge/linux-64::libglib-2.80.2-hf974151_0 
      libpng             conda-forge/linux-64::libpng-1.6.43-h2797004_0 
      libxcb             conda-forge/linux-64::libxcb-1.15-h0b41bf4_0 
      openbabel          conda-forge/linux-64::openbabel-3.1.1-py310hbff9852_9 
      pcre2              conda-forge/linux-64::pcre2-10.43-hcad00b1_0 
      pixman             conda-forge/linux-64::pixman-0.43.2-h59595ed_0 
      pthread-stubs      conda-forge/linux-64::pthread-stubs-0.4-h36c2ea0_1001 
      xorg-kbproto       conda-forge/linux-64::xorg-kbproto-1.0.7-h7f98852_1002 
      xorg-libice        conda-forge/linux-64::xorg-libice-1.1.1-hd590300_0 
      xorg-libsm         conda-forge/linux-64::xorg-libsm-1.2.4-h7391055_0 
      xorg-libx11        conda-forge/linux-64::xorg-libx11-1.8.9-h8ee46fc_0 
      xorg-libxau        conda-forge/linux-64::xorg-libxau-1.0.11-hd590300_0 
      xorg-libxdmcp      conda-forge/linux-64::xorg-libxdmcp-1.1.3-h7f98852_0 
      xorg-libxext       conda-forge/linux-64::xorg-libxext-1.3.4-h0b41bf4_2 
      xorg-libxrender    conda-forge/linux-64::xorg-libxrender-0.9.11-hd590300_0 
      xorg-renderproto   conda-forge/linux-64::xorg-renderproto-0.11.1-h7f98852_1002 
      xorg-xextproto     conda-forge/linux-64::xorg-xextproto-7.3.0-h0b41bf4_1003 
      xorg-xproto        conda-forge/linux-64::xorg-xproto-7.0.31-h7f98852_1007 
      zlib               conda-forge/linux-64::zlib-1.2.13-hd590300_5 
    
    The following packages will be UPDATED:
    
      ca-certificates                     2023.11.17-hbcca054_0 --> 2024.2.2-hbcca054_0 
      certifi                           2023.11.17-pyhd8ed1ab_0 --> 2024.2.2-pyhd8ed1ab_0 
      openssl                                  3.2.0-hd590300_1 --> 3.3.0-hd590300_0 
    
    
    
    Downloading and Extracting Packages:
    openbabel-3.1.1      | 5.0 MB    | :   0% 0/1 [00:00<?, ?it/s]
    libglib-2.80.2       | 3.7 MB    | :   0% 0/1 [00:00<?, ?it/s][A
    
    openssl-3.3.0        | 2.8 MB    | :   0% 0/1 [00:00<?, ?it/s][A[A
    
    
    font-ttf-ubuntu-0.83 | 1.5 MB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A
    
    
    
    cairo-1.18.0         | 959 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A
    
    
    
    
    pcre2-10.43          | 929 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A
    
    
    
    
    
    xorg-libx11-1.8.9    | 809 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A
    
    
    
    
    
    
    font-ttf-source-code | 684 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A
    
    
    
    
    
    
    
    freetype-2.12.1      | 620 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    font-ttf-dejavu-sans | 388 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    pixman-0.43.2        | 378 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    libxcb-1.15          | 375 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    libpng-1.6.43        | 281 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    fontconfig-2.14.2    | 266 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    certifi-2024.2.2     | 157 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ca-certificates-2024 | 152 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    expat-2.6.2          | 134 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    font-ttf-inconsolata | 94 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    zlib-1.2.13          | 91 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     ... (more hidden) ...[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    libglib-2.80.2       | 3.7 MB    | :   0% 0.004187418677717254/1 [00:00<00:29, 30.03s/it][A
    
    
    openbabel-3.1.1      | 5.0 MB    | :   0% 0.0031211275225574045/1 [00:00<00:44, 44.78s/it]
    
    openssl-3.3.0        | 2.8 MB    | :   1% 0.005659047239435657/1 [00:00<00:25, 25.20s/it][A[A
    
    
    
    cairo-1.18.0         | 959 KB    | :   2% 0.016678356310524445/1 [00:00<00:07,  8.05s/it][A[A[A[A
    openbabel-3.1.1      | 5.0 MB    | :  53% 0.5337128063573161/1 [00:00<00:00,  2.63it/s]   
    
    
    
    
    pcre2-10.43          | 929 KB    | :   2% 0.017230953034505024/1 [00:00<00:15, 15.43s/it][A[A[A[A[A
    
    
    
    
    
    xorg-libx11-1.8.9    | 809 KB    | :   2% 0.01978600584498708/1 [00:00<00:13, 13.88s/it][A[A[A[A[A[A
    
    
    
    
    
    
    font-ttf-source-code | 684 KB    | :   2% 0.02337852839697837/1 [00:00<00:14, 14.87s/it][A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    font-ttf-dejavu-sans | 388 KB    | :   4% 0.04123109444598234/1 [00:00<00:08,  8.53s/it][A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    freetype-2.12.1      | 620 KB    | :   3% 0.025802712560553852/1 [00:00<00:13, 13.90s/it][A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    pixman-0.43.2        | 378 KB    | :   4% 0.042354960628292825/1 [00:00<00:08,  8.64s/it][A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    libxcb-1.15          | 375 KB    | :   4% 0.042640238602116395/1 [00:00<00:09,  9.62s/it][A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    libpng-1.6.43        | 281 KB    | :   6% 0.05684526804084366/1 [00:00<00:07,  7.45s/it][A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    fontconfig-2.14.2    | 266 KB    | :   6% 0.060233079666188745/1 [00:00<00:06,  7.35s/it][A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    font-ttf-ubuntu-0.83 | 1.5 MB    | : 100% 1.0/1 [00:00<00:00,  2.36it/s]                 [A[A[A
    
    
    font-ttf-ubuntu-0.83 | 1.5 MB    | : 100% 1.0/1 [00:00<00:00,  2.36it/s][A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    certifi-2024.2.2     | 157 KB    | :  10% 0.10204348557228184/1 [00:00<00:03,  4.44s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ca-certificates-2024 | 152 KB    | :  11% 0.10540943949765814/1 [00:00<00:03,  4.30s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    expat-2.6.2          | 134 KB    | :  12% 0.11904640804493305/1 [00:00<00:03,  3.84s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    font-ttf-inconsolata | 94 KB     | :  17% 0.16972961773541903/1 [00:00<00:02,  2.76s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    cairo-1.18.0         | 959 KB    | : 100% 1.0/1 [00:00<00:00,  2.23it/s]                 [A[A[A[A
    
    
    
    cairo-1.18.0         | 959 KB    | : 100% 1.0/1 [00:00<00:00,  2.23it/s][A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    zlib-1.2.13          | 91 KB     | :  18% 0.1765041745219499/1 [00:00<00:02,  2.82s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     ... (more hidden) ...[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    openssl-3.3.0        | 2.8 MB    | : 100% 1.0/1 [00:00<00:00,  1.28it/s]                 [A[A
    
    openssl-3.3.0        | 2.8 MB    | : 100% 1.0/1 [00:00<00:00,  1.28it/s][A[A
    
    
    
    
    
    xorg-libx11-1.8.9    | 809 KB    | : 100% 1.0/1 [00:00<00:00,  1.09it/s]                [A[A[A[A[A[A
    
    
    
    
    
    xorg-libx11-1.8.9    | 809 KB    | : 100% 1.0/1 [00:00<00:00,  1.09it/s][A[A[A[A[A[A
    libglib-2.80.2       | 3.7 MB    | : 100% 1.0/1 [00:01<00:00,  4.06it/s]               [A
    
    
    
    
    
    
    
    
    font-ttf-dejavu-sans | 388 KB    | : 100% 1.0/1 [00:01<00:00,  1.08s/it]                [A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    font-ttf-dejavu-sans | 388 KB    | : 100% 1.0/1 [00:01<00:00,  1.08s/it][A[A[A[A[A[A[A[A[A
    
    
    
    
    pcre2-10.43          | 929 KB    | : 100% 1.0/1 [00:01<00:00,  1.13s/it]                 [A[A[A[A[A
    
    
    
    
    pcre2-10.43          | 929 KB    | : 100% 1.0/1 [00:01<00:00,  1.13s/it][A[A[A[A[A
    
    
    
    
    
    
    
    
    
    pixman-0.43.2        | 378 KB    | : 100% 1.0/1 [00:01<00:00,  1.14s/it]                 [A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    pixman-0.43.2        | 378 KB    | : 100% 1.0/1 [00:01<00:00,  1.14s/it][A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    freetype-2.12.1      | 620 KB    | : 100% 1.0/1 [00:01<00:00,  1.18s/it]                 [A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    freetype-2.12.1      | 620 KB    | : 100% 1.0/1 [00:01<00:00,  1.18s/it][A[A[A[A[A[A[A[A
    
    
    
    
    
    
    font-ttf-source-code | 684 KB    | : 100% 1.0/1 [00:01<00:00,  1.42s/it]                [A[A[A[A[A[A[A
    
    
    
    
    
    
    font-ttf-source-code | 684 KB    | : 100% 1.0/1 [00:01<00:00,  1.42s/it][A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    libxcb-1.15          | 375 KB    | : 100% 1.0/1 [00:01<00:00,  1.53s/it]                 [A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    libxcb-1.15          | 375 KB    | : 100% 1.0/1 [00:01<00:00,  1.53s/it][A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    libpng-1.6.43        | 281 KB    | : 100% 1.0/1 [00:01<00:00,  1.57s/it]                [A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    openbabel-3.1.1      | 5.0 MB    | : 100% 1.0/1 [00:01<00:00,  1.96s/it]
    
    
    
    
    
    
    
    
    
    
    
    
    fontconfig-2.14.2    | 266 KB    | : 100% 1.0/1 [00:01<00:00,  1.65s/it]                 [A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ca-certificates-2024 | 152 KB    | : 100% 1.0/1 [00:01<00:00,  1.67s/it]                [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    fontconfig-2.14.2    | 266 KB    | : 100% 1.0/1 [00:01<00:00,  1.65s/it][A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ca-certificates-2024 | 152 KB    | : 100% 1.0/1 [00:01<00:00,  1.67s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    certifi-2024.2.2     | 157 KB    | : 100% 1.0/1 [00:01<00:00,  1.71s/it]                [A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    certifi-2024.2.2     | 157 KB    | : 100% 1.0/1 [00:01<00:00,  1.71s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    expat-2.6.2          | 134 KB    | : 100% 1.0/1 [00:01<00:00,  1.72s/it]                [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    expat-2.6.2          | 134 KB    | : 100% 1.0/1 [00:01<00:00,  1.72s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    zlib-1.2.13          | 91 KB     | : 100% 1.0/1 [00:01<00:00,  1.76s/it]               [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    zlib-1.2.13          | 91 KB     | : 100% 1.0/1 [00:01<00:00,  1.76s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    font-ttf-inconsolata | 94 KB     | : 100% 1.0/1 [00:01<00:00,  1.78s/it]                [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    font-ttf-inconsolata | 94 KB     | : 100% 1.0/1 [00:01<00:00,  1.78s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     ... (more hidden) ...[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     ... (more hidden) ...[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                                                                            
                                                                            [A
    
                                                                            [A[A
    
    
                                                                            [A[A[A
    
    
    
                                                                            [A[A[A[A
    
    
    
    
                                                                            [A[A[A[A[A
    
    
    
    
    
                                                                            [A[A[A[A[A[A
    
    
    
    
    
    
                                                                            [A[A[A[A[A[A[A
    
    
    
    
    
    
    
                                                                            [A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
                                                                            [A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
                                                                            [A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
                                                                            [A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
                                                                            [A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
                                                                            [A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
                                                                            [A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                                                                            [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                                                                            [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                                                                            [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                                                                            [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
    [A
    
    [A[A
    
    
    [A[A[A
    
    
    
    [A[A[A[A
    
    
    
    
    [A[A[A[A[A
    
    
    
    
    
    [A[A[A[A[A[A
    
    
    
    
    
    
    [A[A[A[A[A[A[A
    
    
    
    
    
    
    
    [A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    [A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    [A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    [A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    [A[A[A[A[A[A[A[A[A[A[A[A
    
    
    
    
    
    
    
    
    
    
    
    
    [A[A[A[A[A[A[A[A[A[A[A[A[A
    Preparing transaction: / done
    Verifying transaction: \ | / - \ done
    Executing transaction: / - \ | / - \ | done
    Channels:
     - conda-forge
    Platform: linux-64
    Collecting package metadata (repodata.json): - \ | / - \ | / done
    Solving environment: \ | / - \ | done
    
    
    ==> WARNING: A newer version of conda exists. <==
        current version: 23.11.0
        latest version: 24.4.0
    
    Please update conda by running
    
        $ conda update -n base -c conda-forge conda
    
    
    
    ## Package Plan ##
    
      environment location: /usr/local
    
      added / updated specs:
        - conda-forge::cpulimit
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        cpulimit-0.2               |    h14c3975_1000          15 KB  conda-forge
        ------------------------------------------------------------
                                               Total:          15 KB
    
    The following NEW packages will be INSTALLED:
    
      cpulimit           conda-forge/linux-64::cpulimit-0.2-h14c3975_1000 
    
    
    
    Downloading and Extracting Packages:
                                                                            
    Preparing transaction: - done
    Verifying transaction: | done
    Executing transaction: - done
    

.. code:: python

    from ChemicalDice import smiles_preprocess
    from ChemicalDice import bioactivity
    from ChemicalDice import chemberta
    from ChemicalDice import Grover
    from ChemicalDice import ImageMol
    from ChemicalDice import chemical
    from ChemicalDice import quantum


.. parsed-literal::

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:88: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    


.. parsed-literal::

    config.json:   0%|          | 0.00/631 [00:00<?, ?B/s]



.. parsed-literal::

    pytorch_model.bin:   0%|          | 0.00/13.7M [00:00<?, ?B/s]


.. parsed-literal::

    Some weights of RobertaModel were not initialized from the model checkpoint at DeepChem/ChemBERTa-77M-MLM and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    


.. parsed-literal::

    tokenizer_config.json:   0%|          | 0.00/1.27k [00:00<?, ?B/s]



.. parsed-literal::

    vocab.json:   0%|          | 0.00/6.96k [00:00<?, ?B/s]



.. parsed-literal::

    merges.txt:   0%|          | 0.00/52.0 [00:00<?, ?B/s]



.. parsed-literal::

    added_tokens.json:   0%|          | 0.00/25.0 [00:00<?, ?B/s]



.. parsed-literal::

    special_tokens_map.json:   0%|          | 0.00/420 [00:00<?, ?B/s]



.. parsed-literal::

    tokenizer.json:   0%|          | 0.00/8.26k [00:00<?, ?B/s]


.. parsed-literal::

    [WARNING] Horovod cannot be imported; multi-GPU training is unsupported
    

.. code:: python

    # download prerequisites for quantum,
    quantum.get_mopac_prerequisites()


.. parsed-literal::

    Mopac is downloaded
    Morse is compiled
    

.. code:: python

    input_file = "freesolv.csv"
    import os
    os.mkdir("data")

.. code:: python

    smiles_preprocess.add_canonical_smiles(input_file)
    smiles_preprocess.create_mol2_files(input_file, output_dir = "data_mol2files", ncpu=10)
    smiles_preprocess.create_sdf_files(input_file, output_dir = "data_sdffiles")


.. parsed-literal::

    /usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()
    100%|██████████| 21/21 [00:17<00:00,  1.20it/s]
    

.. parsed-literal::

    making directory  data_sdffiles
    

.. parsed-literal::

    21it [00:00, 612.33it/s]
    

.. code:: python

    quantum.descriptor_calculator(input_file,, output_file="data/mopac.csv", ncpu=5)
    Grover.get_embeddings(input_file,  output_file_name="data/Grover.csv")
    
    ImageMol.image_to_embeddings(input_file, output_file_name="data/ImageMol.csv")


::


    ---------------------------------------------------------------------------

    SyntaxError                               Traceback (most recent call last)

    /usr/local/lib/python3.10/dist-packages/IPython/core/compilerop.py in ast_parse(self, source, filename, symbol)
         99         Arguments are exactly the same as ast.parse (in the standard library),
        100         and are passed to the built-in compile function."""
    --> 101         return compile(source, filename, symbol, self.flags | PyCF_ONLY_AST, 1)
        102 
        103     def reset_compiler_flags(self):
    

    SyntaxError: invalid syntax (<ipython-input-5-37f26cf30edd>, line 1)


.. code:: python

    quantum.descriptor_calculator(input_file, output_dir="data_mopfiles", output_file="data/mopac.csv", ncpu=5)

.. code:: python

    Grover.get_embeddings(input_file, output_dir = "data_grover", output_file_name="data/Grover.csv", model_checkpoint_file_path ="grover_large.pt")


.. parsed-literal::

    making directory  data_grover
    

.. parsed-literal::

    100%|██████████| 642/642 [00:36<00:00, 17.41it/s]
    Total size = 642
    Generating...
    

.. parsed-literal::

    Loading data
    

.. parsed-literal::

    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.0.mpn_q.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.0.mpn_q.W_h.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.0.mpn_k.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.0.mpn_k.W_h.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.0.mpn_v.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.0.mpn_v.W_h.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.1.mpn_q.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.1.mpn_q.W_h.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.1.mpn_k.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.1.mpn_k.W_h.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.1.mpn_v.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.1.mpn_v.W_h.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.2.mpn_q.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.2.mpn_q.W_h.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.2.mpn_k.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.2.mpn_k.W_h.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.2.mpn_v.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.2.mpn_v.W_h.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.3.mpn_q.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.3.mpn_q.W_h.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.3.mpn_k.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.3.mpn_k.W_h.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.3.mpn_v.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.heads.3.mpn_v.W_h.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.act_func.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.layernorm.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.layernorm.bias".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.W_i.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.attn.linear_layers.0.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.attn.linear_layers.0.bias".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.attn.linear_layers.1.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.attn.linear_layers.1.bias".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.attn.linear_layers.2.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.attn.linear_layers.2.bias".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.attn.output_linear.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.W_o.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.sublayer.norm.weight".
    Loading pretrained parameter "grover.encoders.edge_blocks.0.sublayer.norm.bias".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.0.mpn_q.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.0.mpn_q.W_h.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.0.mpn_k.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.0.mpn_k.W_h.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.0.mpn_v.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.0.mpn_v.W_h.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.1.mpn_q.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.1.mpn_q.W_h.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.1.mpn_k.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.1.mpn_k.W_h.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.1.mpn_v.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.1.mpn_v.W_h.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.2.mpn_q.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.2.mpn_q.W_h.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.2.mpn_k.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.2.mpn_k.W_h.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.2.mpn_v.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.2.mpn_v.W_h.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.3.mpn_q.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.3.mpn_q.W_h.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.3.mpn_k.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.3.mpn_k.W_h.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.3.mpn_v.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.heads.3.mpn_v.W_h.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.act_func.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.layernorm.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.layernorm.bias".
    Loading pretrained parameter "grover.encoders.node_blocks.0.W_i.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.attn.linear_layers.0.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.attn.linear_layers.0.bias".
    Loading pretrained parameter "grover.encoders.node_blocks.0.attn.linear_layers.1.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.attn.linear_layers.1.bias".
    Loading pretrained parameter "grover.encoders.node_blocks.0.attn.linear_layers.2.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.attn.linear_layers.2.bias".
    Loading pretrained parameter "grover.encoders.node_blocks.0.attn.output_linear.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.W_o.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.sublayer.norm.weight".
    Loading pretrained parameter "grover.encoders.node_blocks.0.sublayer.norm.bias".
    Loading pretrained parameter "grover.encoders.ffn_atom_from_atom.W_1.weight".
    Loading pretrained parameter "grover.encoders.ffn_atom_from_atom.W_1.bias".
    Loading pretrained parameter "grover.encoders.ffn_atom_from_atom.W_2.weight".
    Loading pretrained parameter "grover.encoders.ffn_atom_from_atom.W_2.bias".
    Loading pretrained parameter "grover.encoders.ffn_atom_from_atom.act_func.weight".
    Loading pretrained parameter "grover.encoders.ffn_atom_from_bond.W_1.weight".
    Loading pretrained parameter "grover.encoders.ffn_atom_from_bond.W_1.bias".
    Loading pretrained parameter "grover.encoders.ffn_atom_from_bond.W_2.weight".
    Loading pretrained parameter "grover.encoders.ffn_atom_from_bond.W_2.bias".
    Loading pretrained parameter "grover.encoders.ffn_atom_from_bond.act_func.weight".
    Loading pretrained parameter "grover.encoders.ffn_bond_from_atom.W_1.weight".
    Loading pretrained parameter "grover.encoders.ffn_bond_from_atom.W_1.bias".
    Loading pretrained parameter "grover.encoders.ffn_bond_from_atom.W_2.weight".
    Loading pretrained parameter "grover.encoders.ffn_bond_from_atom.W_2.bias".
    Loading pretrained parameter "grover.encoders.ffn_bond_from_atom.act_func.weight".
    Loading pretrained parameter "grover.encoders.ffn_bond_from_bond.W_1.weight".
    Loading pretrained parameter "grover.encoders.ffn_bond_from_bond.W_1.bias".
    Loading pretrained parameter "grover.encoders.ffn_bond_from_bond.W_2.weight".
    Loading pretrained parameter "grover.encoders.ffn_bond_from_bond.W_2.bias".
    Loading pretrained parameter "grover.encoders.ffn_bond_from_bond.act_func.weight".
    Loading pretrained parameter "grover.encoders.atom_from_atom_sublayer.norm.weight".
    Loading pretrained parameter "grover.encoders.atom_from_atom_sublayer.norm.bias".
    Loading pretrained parameter "grover.encoders.atom_from_bond_sublayer.norm.weight".
    Loading pretrained parameter "grover.encoders.atom_from_bond_sublayer.norm.bias".
    Loading pretrained parameter "grover.encoders.bond_from_atom_sublayer.norm.weight".
    Loading pretrained parameter "grover.encoders.bond_from_atom_sublayer.norm.bias".
    Loading pretrained parameter "grover.encoders.bond_from_bond_sublayer.norm.weight".
    Loading pretrained parameter "grover.encoders.bond_from_bond_sublayer.norm.bias".
    Loading pretrained parameter "grover.encoders.act_func_node.weight".
    Loading pretrained parameter "grover.encoders.act_func_edge.weight".
    /usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py:558: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(_create_warning_msg(
    /usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()
    /usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()
    

.. code:: python

    ImageMol.add_image_files(input_file, output_dir = "data_imagefiles")


.. parsed-literal::

    making directory  data_imagefiles
    

.. code:: python

    ImageMol.image_to_embeddings(input_file, output_file_name="data/ImageMol.csv", model_checkpoint_file_path='ImageMol.pth.tar')


.. parsed-literal::

    Warning: There's no GPU available on this machine, training will be performed on CPU.
    

.. parsed-literal::

    /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
      warnings.warn(msg)
    /usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()
    

.. code:: python

    chemberta.smiles_to_embeddings(input_file, output_file = "data/Chemberta.csv")

.. code:: python

    #import numpy as np
    #np.bool = np.bool_
    bioactivity.calculate_descriptors(input_file, output_file = "data/Signaturizer.csv")

.. code:: python

    chemical.descriptor_calculator(input_file, output_file="data/mordred.csv")

.. code:: python

    from ChemicalDice.plot_data import plot_model_metrics
    from ChemicalDice.plot_data import plot_models_barplot

.. code:: python

    data_paths = {
        "tabular1": "data/Chemberta_embeddings.csv",
        "tabular2": "data/graph_embeddings.csv",
        "tabular3": "fs_data/mopac_descriptors.csv",
        "tabular4": "fs_data/mordred_descriptors.csv",
        "tabular5": "fs_data/signaturizer_descriptors.csv"
    }
    
    
    fusiondata = fusionData(data_paths = data_paths,label_file_path="freesolv.csv",label_column="labels",id_column="ID")


.. parsed-literal::

    Successfully loaded, processed for 'tabular1'.
    Successfully loaded, processed for 'tabular2'.
    Successfully loaded, processed for 'tabular3'.
    Successfully loaded, processed for 'tabular4'.
    Successfully loaded, processed for 'tabular5'.
    

.. code:: python

    data_paths = {
        "tabular1": "fs_data/Chemberta_embeddings.csv",
        "tabular2": "fs_data/graph_embeddings.csv",
        "tabular3": "fs_data/mopac_descriptors.csv",
        "tabular4": "fs_data/mordred_descriptors.csv",
        "tabular5": "fs_data/signaturizer_descriptors.csv",
        "tabular6": "fs_data/ImageMol_embeddings.csv"
    }
    
    
    fusiondata = fusionData(data_paths = data_paths,label_file_path="freesolv.csv",label_column="labels",id_column="ID")
    fusiondata.keep_common_samples()
    fusiondata.remove_empty_features()
    fusiondata.ImputeData(method="knn",class_specific=False)
    fusiondata.scale_data(scaling_type = "standardize")


.. parsed-literal::

    Successfully loaded, processed for 'tabular1'.
    Successfully loaded, processed for 'tabular2'.
    Successfully loaded, processed for 'tabular3'.
    Successfully loaded, processed for 'tabular4'.
    Successfully loaded, processed for 'tabular5'.
    Successfully loaded, processed for 'tabular6'.
    Imputation Done
    

.. code:: python

    # evaluate all models
    fusiondata.evaluate_fusion_models_nfold(n_components=4096,
                                              methods= ["AER",'pca', 'plsda'],# 'ica', 'ipca', 'cca', 'tsne', 'kpca', 'rks', 'SEM', 'autoencoder', 'tensordecompose', 'plsda'],
                                              n_folds = 10,
                                              regression = True)


.. parsed-literal::

    The directory comaprision_data_fold_0_of_10/training_data/ was created.
    The directory comaprision_data_fold_0_of_10/testing_data/ was created.
    Method name AER
    384
    5000
    212
    1422
    3200
    10000
    k= 4096
    Training AUTOENCODER finished Train loss: 0.706456493247639 val loss: 0
    (642,)
    Done
    384
    5000
    212
    1422
    3200
    10000
    (577,)
    Done
    Training data is fused.
    384
    5000
    212
    1422
    3200
    10000
    (65,)
    Done
    Testing data is fused. 
    Method name pca
    

::


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-21-d3d579909cb6> in <cell line: 2>()
          1 # evaluate all models
    ----> 2 fusiondata.evaluate_fusion_models_nfold(n_components=4096,
          3                                           methods= ["AER",'pca', 'plsda'],# 'ica', 'ipca', 'cca', 'tsne', 'kpca', 'rks', 'SEM', 'autoencoder', 'tensordecompose', 'plsda'],
          4                                           n_folds = 10,
          5                                           regression = True)
    

    /usr/local/lib/python3.10/dist-packages/ChemicalDice/fusionData.py in evaluate_fusion_models_nfold(self, n_components, n_folds, methods, regression, **kwargs)
       1186                     # Fuse all data in dataframes
       1187                     print("Method name", method_chemdice)
    -> 1188                     self.fuseFeaturesTrain(n_components=n_components, method=method_chemdice)
       1189                     X_train = self.fusedData_train
       1190                     y_train = self.train_label
    

    /usr/local/lib/python3.10/dist-packages/ChemicalDice/fusionData.py in fuseFeaturesTrain(self, n_components, method, **kwargs)
        586         if method in ['pca', 'ica', 'ipca']:
        587             merged_df = pd.concat(df_list, axis=1)
    --> 588             fused_df1 = apply_analysis_linear1(merged_df, analysis_type=method, n_components=n_components, **kwargs)
        589         elif method in ['cca']:
        590             all_combinations = []
    

    /usr/local/lib/python3.10/dist-packages/ChemicalDice/analyse_data.py in apply_analysis_linear1(data, analysis_type, n_components, **kwargs)
        339     if analysis_type.lower() == 'pca':
        340         pca = PCA(n_components=n_components, **kwargs)
    --> 341         transformed_data = pd.DataFrame(pca.fit_transform(data),
        342                                          columns=[f'PC{i+1}' for i in range(pca.n_components_)],index = data.index)
        343     elif analysis_type.lower() == 'ica':
    

    /usr/local/lib/python3.10/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
        138     @wraps(f)
        139     def wrapped(self, X, *args, **kwargs):
    --> 140         data_to_wrap = f(self, X, *args, **kwargs)
        141         if isinstance(data_to_wrap, tuple):
        142             # only wrap the first output for cross decomposition
    

    /usr/local/lib/python3.10/dist-packages/sklearn/decomposition/_pca.py in fit_transform(self, X, y)
        460         self._validate_params()
        461 
    --> 462         U, S, Vt = self._fit(X)
        463         U = U[:, : self.n_components_]
        464 
    

    /usr/local/lib/python3.10/dist-packages/sklearn/decomposition/_pca.py in _fit(self, X)
        510         # Call different fits for either full or truncated SVD
        511         if self._fit_svd_solver == "full":
    --> 512             return self._fit_full(X, n_components)
        513         elif self._fit_svd_solver in ["arpack", "randomized"]:
        514             return self._fit_truncated(X, n_components, self._fit_svd_solver)
    

    /usr/local/lib/python3.10/dist-packages/sklearn/decomposition/_pca.py in _fit_full(self, X, n_components)
        524                 )
        525         elif not 0 <= n_components <= min(n_samples, n_features):
    --> 526             raise ValueError(
        527                 "n_components=%r must be between 0 and "
        528                 "min(n_samples, n_features)=%r with "
    

    ValueError: n_components=4096 must be between 0 and min(n_samples, n_features)=577 with svd_solver='full'


.. code:: python

    top_models = fusiondata.Accuracy_metrics.iloc[0:200,:]
    plot_model_boxplot(top_models, save_dir = "output_plots_freesolv")

.. code:: python

    fusiondata.evaluate_fusion_models_scaffold_split( methods= ["AER","pca",'plsda'],
                                              AER_dim = 256,
                                              regression = True,
                                              split_type = "random")

.. code:: python

    matrics = fusiondata.scaffold_split_result
    plot_models_barplot(matrics,save_dir = "output_plots_freesolv_scaffold_splitting")






.. code:: python

    !cp ChemicalDice/*.py /content/drive/MyDrive/ChemicalDice/ChemicalDice/

.. code:: python

    !ls ChemicalDice/ | wc -l


.. parsed-literal::

    54
    

.. code:: python

    !ls /content/drive/MyDrive/ChemicalDice/ChemicalDice/ | wc -l


.. parsed-literal::

    52
    

.. code:: python

    !python3 -c 'import keras; print(keras.__version__)'
    # 2.15.0


.. parsed-literal::

    2024-05-03 17:42:22.727679: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-05-03 17:42:23.851630: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/usr/local/lib/python3.10/site-packages/keras/__init__.py", line 3, in <module>
        from keras import __internal__
      File "/usr/local/lib/python3.10/site-packages/keras/__internal__/__init__.py", line 3, in <module>
        from keras.__internal__ import backend
      File "/usr/local/lib/python3.10/site-packages/keras/__internal__/backend/__init__.py", line 3, in <module>
        from keras.src.backend import _initialize_variables as initialize_variables
      File "/usr/local/lib/python3.10/site-packages/keras/src/__init__.py", line 21, in <module>
        from keras.src import applications
      File "/usr/local/lib/python3.10/site-packages/keras/src/applications/__init__.py", line 18, in <module>
        from keras.src.applications.convnext import ConvNeXtBase
      File "/usr/local/lib/python3.10/site-packages/keras/src/applications/convnext.py", line 33, in <module>
        from keras.src.engine import sequential
      File "/usr/local/lib/python3.10/site-packages/keras/src/engine/sequential.py", line 24, in <module>
        from keras.src.engine import functional
      File "/usr/local/lib/python3.10/site-packages/keras/src/engine/functional.py", line 33, in <module>
        from keras.src.engine import training as training_lib
      File "/usr/local/lib/python3.10/site-packages/keras/src/engine/training.py", line 48, in <module>
        from keras.src.saving import saving_api
      File "/usr/local/lib/python3.10/site-packages/keras/src/saving/saving_api.py", line 25, in <module>
        from keras.src.saving.legacy import save as legacy_sm_saving_lib
      File "/usr/local/lib/python3.10/site-packages/keras/src/saving/legacy/save.py", line 27, in <module>
        from keras.src.saving.legacy.saved_model import load_context
      File "/usr/local/lib/python3.10/site-packages/keras/src/saving/legacy/saved_model/load_context.py", line 68, in <module>
        tf.__internal__.register_load_context_function(in_load_context)
    AttributeError: module 'tensorflow._api.v2.compat.v2.__internal__' has no attribute 'register_load_context_function'. Did you mean: 'register_call_context_function'?
    

.. code:: python

    #calculate chemical descriptors from given sdf files path from csv file
    chemical.descriptor_calculator(input_file, output_file="data/mordred.csv")  # C784 C785


.. parsed-literal::

    642it [02:00,  5.35it/s]
    

.. code:: python

    
    from ChemicalDice import smiles_preprocess
    from ChemicalDice import chemberta
    from ChemicalDice import bioactivity
    from ChemicalDice import Grover
    from ChemicalDice import ImageMol
    from ChemicalDice import chemical
    from ChemicalDice import quantum
    
    
    
    
    smiles_preprocess.add_canonical_smiles(input_file)
    smiles_preprocess.create_mol2_files(input_file, output_dir = "data_mol2files", ncpu=10)
    smiles_preprocess.create_sdf_files(input_file, output_dir = "data_sdffiles")
    
    
    
    #calculate chemical descriptors from given sdf files path from csv file
    chemical.descriptor_calculator(input_file, output_file="data/mordred.csv")  # C784 C785
    
    
    
    
    
    
    
    
    
    
    
    from ChemicalDice.fusionData import fusionData
    from ChemicalDice.plot_data import plot_model_metrics
    from ChemicalDice.plot_data import plot_model_boxplot
    from ChemicalDice.plot_data import plot_metrics
    from ChemicalDice.getEmbeddings import AutoencoderReconstructor
    
    
    data_paths = {
        "tabular1": "data/Chemberta_embeddings.csv",
        "tabular2": "data/graph_embeddings.csv",
        "tabular3": "data/mopac_descriptors.csv",
        "tabular4": "data/mordred_descriptors.csv",
        "tabular5": "data/signaturizer_descriptors.csv",
        "tabular6": "data/ImageMol_embeddings.csv"
    }
    
    
    fusiondata = fusionData(data_paths = data_paths)
    
    import csv
    
    # Open the file in read mode
    with open('ChemicalDice/cc.csv', 'r') as file:
        reader = csv.reader(file)
        # Read the first row and convert it to a list
        common_columns = list(next(reader))
    
    
    
    
    from sklearn.impute import KNNImputer
    import pandas as pd
    import numpy as np
    
    
    import pandas as pd
    from sklearn.impute import KNNImputer
    
    
    
    
    
    
    # Assuming df is your DataFrame and 'class' is your class column
    def impute_class_specific(df):
        # Create a copy of the DataFrame for each class
        # Create a copy of the DataFrame for each class
        df_A = df.copy()
        # Create the KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        # Perform KNN imputation for each class
        df_A.iloc[:, :] = imputer.fit_transform(df_A.iloc[:, :])
        # Concatenate the imputed DataFrames
        df_imputed = df_A
        return df_imputed
    
    df = fusiondata.dataframes['tabular4']
    
    df2 = df.drop(common_columns, axis =1)
    # imputer = KNNImputer(n_neighbors=5)
    # df2_imputed = imputer.fit_transform(df2)
    # nan_count = np.count_nonzero(np.isnan(df2_imputed))
    
    df_impudted= impute_class_specific(df2)
    
    fusiondata.dataframes['tabular4']  = df_impudted
    
    
    
    # df = fusiondata.dataframes['tabular3']
    
    
    # imputer = KNNImputer(n_neighbors=5)
    # df2_imputed = imputer.fit_transform(df2)
    # nan_count = np.count_nonzero(np.isnan(df2_imputed))
    
    # df_impudted= impute_class_specific(df)
    
    # fusiondata.dataframes['tabular3']  = df_impudted
    
    
    fusiondata.keep_common_samples()
    
    #fusiondata.scale_data(scaling_type = 'standardize')
    
    
    
    all_embeddings = AutoencoderReconstructor(fusiondata.dataframes['tabular3'],
                             fusiondata.dataframes['tabular1'],
                             fusiondata.dataframes['tabular4'],
                             fusiondata.dataframes['tabular5'],
                             fusiondata.dataframes['tabular6'],
                             fusiondata.dataframes['tabular2'],
                             embedding_sizes=[256])
    
    
    all_embeddings
