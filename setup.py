from setuptools import setup,find_packages

with open('README.md', 'r') as f:
    README = f.read()

setup(
    name='talk2doc',
    version='0.0.1',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Sukhbinder Singh',
    author_email='sukh2010@yahoo.com',
    description='A tool to ask questions to documents in a set of PDFs.',
    entry_points={
        'console_scripts': ['talk2doc=talk2doc.talk2doc:main'],
    },
    packages=find_packages(),
    install_requires=[
        'langchain_community',
        'pypdf',
        'langchain',
        'langchain-core',
        'ollama',
        'tiktoken',
    ],
)
