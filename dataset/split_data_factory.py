from dataset import dataloader

def split_dataloader(split_type=None,dataset=None,args_dataloader=None,seed=None):
    assert dataset in ["CWRU", "LASPI"], f"Invalid dataset: {dataset}"
    assert split_type in ["independent", "speed_stratified", "speed_load_stratified", "sample_stratified"] , f"Invalid split type: {split_type}"
    
    # Split dataloaders - sampling spectrum independently
    if split_type == "independent":
        train_loader, valid_loader, test_loader, labels = dataloader.get_homogenous_split_dataloaders(args_dataloader,seed=seed)

    # Split dataloaders - speed stratified sampling
    elif split_type == "speed_stratified":

        if dataset == "CWRU":
            # Speeds: 1730,1750, 1772 and 1797 RPM for CWRU
            train_loader, valid_loader, test_loader, labels = dataloader.get_speed_stratified_dataloaders(
                args=args_dataloader,
                train_val_speeds=['1750', '1772', '1797'],
                test_speeds=['1730'],
                seed=seed,
            )
        
        elif dataset == "LASPI":
            # Speeds: 25, 35 and 45 Hz for LASPI
            train_loader, valid_loader, test_loader, labels = dataloader.get_speed_stratified_dataloaders(
                args=args_dataloader,
                train_val_speeds=[25, 45],
                test_speeds=[35],
                seed=seed,
            )

    # Split dataloaders - speed-load stratified sampling
    elif split_type == "speed_load_stratified":

        if dataset == "CWRU":
            # CWRU doesn't have load conditions, so speed-load stratified sampling is not applicable
            raise NotImplementedError("Speed-load stratified sampling not implemented for CWRU yet")

        elif dataset == "LASPI":
            # Loads : 0, 25, 50 and 75 % of full load for LASPI
            # Speeds : 25, 35 and 45 Hz for LASPI
            # Total combinations : 12 (all combinations of 4 loads and 3 speeds) - test on (25Hz, 50% load)
            train_loader, valid_loader, test_loader, labels = dataloader.get_speed_load_stratified_dataloaders(
                args=args_dataloader, 
                train_val_combinations=[(25, 0), (25, 25), (25, 75),
                                        (35, 0), (35, 25), (35, 50), (35, 75),
                                        (45, 0), (45, 25), (45, 50), (45, 75),],
                test_combinations=[(25, 50)],
                seed=seed,
            )
    
    elif split_type == "sample_stratified":

        if dataset == "CWRU":   
            # Not realease yet
            raise NotImplementedError("Sample stratified sampling not implemented for CWRU yet")
        
        elif dataset == "LASPI":
            # Mixed speed train/val, test on fixed speeds and load conditions, but only 1 load condition per speed to avoid overfitting to load conditions
            train_loader, valid_loader, test_loader, labels = dataloader.get_laspi_acquisition_split_dataloaders(
                args=args_dataloader,
                seed=seed,
            )

    return train_loader, valid_loader, test_loader, labels
