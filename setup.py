from io import open
from setuptools import find_packages, setup

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name="qurator-sbb-ner",
    version="0.0.1",
    author="The Qurator Team",
    author_email="qurator@sbb.spk-berlin.de",
    description="Qurator",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='qurator',
    license='Apache',
    url="https://qurator.ai",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=install_requires,
    entry_points={
      'console_scripts': [
        "compile_europeana_historic=qurator.sbb_ner.ground_truth.europeana_historic:main",
        "compile_germ_eval=qurator.sbb_ner.ground_truth.germeval:main",
        "compile_conll=qurator.sbb_ner.ground_truth.conll:main",
        "compile_wikiner=qurator.sbb_ner.ground_truth.wikiner:main",
        "join-gt=qurator.sbb_ner.ground_truth.join_gt:main",
        "bert-ner=qurator.sbb_ner.models.bert:main",

        "collectcorpus=qurator.sbb_ner.models.corpus:collect",
        "bert-pregenerate-trainingdata=qurator.sbb_ner.models.pregenerate_training_data:main",
        "bert-finetune=qurator.sbb_ner.models.finetune_on_pregenerated:main"
      ]
    },
    python_requires='>=3.6.0',
    tests_require=['pytest'],
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)