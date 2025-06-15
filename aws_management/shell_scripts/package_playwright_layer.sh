#!/bin/bash

# Create a temporary directory
mkdir -p temp_layer/python

# Install Playwright and its dependencies
pip install playwright -t temp_layer/python/
pip install playwright-core -t temp_layer/python/

# Download Chrome for AWS Lambda
cd temp_layer/python
python -m playwright install chromium

# Create the layer package
cd ../..
zip -r playwright_layer.zip temp_layer/

# Clean up
rm -rf temp_layer

echo "Playwright layer package created: playwright_layer.zip" 