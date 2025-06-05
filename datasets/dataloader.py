import importlib
import inspect

DATASET_PATHS = {
    "CWRU": "data/raw/CWRU",
    "LASPI": "data/raw/LASPI",
}

def get_dataset(name, **kwargs):
    """
    Returns an instance of the specified dataset class.
    Args:
        name (str): Name of the dataset (e.g., 'CWRU', 'LASPI').
        **kwargs: Additional keyword arguments to pass to the dataset constructor.
    """
    base_path = DATASET_PATHS[name.upper()]
    module = importlib.import_module(f"datasets.{name.lower()}_dataset")
    dataset_class = getattr(module, f"{name.upper()}Dataset")
    # Inspecte la signature du constructeur
    sig = inspect.signature(dataset_class.__init__)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return dataset_class(data_dir=base_path, **valid_args)