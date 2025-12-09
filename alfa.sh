for alfa in 0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1; do
    python train.py method.parameters.alfa=$alfa
done