# Contributing

This is a research repository. Contributions are welcome!

## Guidelines

1. **Code Style**: Follow PEP 8
   ```bash
   black .
   isort .
   ```

2. **Testing**: Run tests before submitting
   ```bash
   pytest tests/
   ```

3. **Documentation**: Update docstrings and README for new features

4. **Commits**: Use clear, descriptive commit messages
   ```
   git commit -m "Add feature: [description]"
   ```

5. **Reproducibility**: Include seeds and configuration in scripts

## Reporting Issues

Please include:
- Python version and OS
- Full error traceback
- Minimal reproducible example
- Your hardware (GPU type, CPU, RAM)

## Research Contributions

For research contributions (new SSL methods, datasets):
- Add your implementation to `models/ssl/` or `dataset/`
- Include docstrings explaining the method
- Add an experiment script in `experiments/`
- Document differences from original papers
