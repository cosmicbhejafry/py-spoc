for anyone working with jupyter notebooks, use nbstripout to make sure local outputs of jupyter notebooks are not published to the main github

initial setup within your conda env:
```
conda install nbstripout
nbstripout --install
```
the filters are stored in the ```.gitattributes``` file.
