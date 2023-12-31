# Climate Case Search

An easy way to search in the climate case database from the Sabin Center.

To get the complete database: https://climate.law.columbia.edu/

![App Screenshot](artefacts/streamlit-searchengine.png)

# /!\ Warnings
* Search engine might be very slow on streamlit's free hosting.
* This is just a demo of a basic search engine. Please consider it as toy search engine.

# For developers

## App launch
```
streamlit run legalsearch/app.py [-- script args]
```

## Torch CPU installation

https://download.pytorch.org/whl/torch/
https://pytorch.org/cppdocs/installing.html

## Sqlite3 setup

Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0

https://www.sqlite.org/chronology.html
https://docs.trychroma.com/troubleshooting#sqlite

## Thanks to
* https://discuss.streamlit.io/t/new-component-dynamic-multi-select-filters/49595
* https://github.com/arsentievalex/streamlit-dynamic-filters/tree/main

