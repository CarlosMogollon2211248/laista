import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
import numpy as np
from sklearn.model_selection import train_test_split # Necesitamos esta función para estratificación

def get_dataloaders(data_dir='./data', batch_size=64, img_size=32):
    """
    Prepara y devuelve los DataLoaders para TRAIN, VAL y TEST,
    limitando el dataset total a 1000 imágenes Y asegurando un BALANCEO ESTRATIFICADO.
    """
    TOTAL_SAMPLES = 10000 
    
    # 1. Definir las transformaciones 
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # 2. Cargar y Concatenar los Datasets
    full_train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset_full = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([full_train_dataset, test_dataset_full])

    # 3. EXTRAER 1000 MUESTRAS DEL CONJUNTO COMPLETO (Selección aleatoria inicial)
    # Obtenemos un Subset de TOTAL_SAMPLES, y el resto lo descartamos.
    indices_pool = np.arange(len(full_dataset))
    
    # Extraer 1000 índices aleatorios del total (usando random.choice o np.random.choice)
    np.random.seed(42)
    selected_indices = np.random.choice(indices_pool, TOTAL_SAMPLES, replace=False)
    
    subset_dataset = Subset(full_dataset, selected_indices)
    subset_targets = []
    for idx in selected_indices:
        subset_targets.append(full_dataset[idx][1])
    
    subset_targets = np.array(subset_targets)

    # 5. DIVISIÓN ESTRATIFICADA (70% Train, 15% Val, 15% Test)
    
    # A. Dividir primero en Train (70%) y Resto (30%)
    train_indices, remaining_indices, _, remaining_targets = train_test_split(
        selected_indices, 
        subset_targets, 
        test_size=0.3, # 300 muestras para Val/Test
        random_state=42, 
        stratify=subset_targets # ¡CRUCIAL! Asegura balanceo
    )


    val_indices, test_indices, _, _ = train_test_split(
        remaining_indices,
        remaining_targets,
        test_size=0.5, # 50% de las 300 restantes = 150 para test
        random_state=42,
        stratify=remaining_targets # ¡CRUCIAL! Asegura balanceo
    )
    
    # 6. Crear los Subsets de PyTorch
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)

    # 7. Creación de DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print(f"Dataset LISTO (Total: {TOTAL_SAMPLES} muestras, ESTRATIFICADO):")
    print(f"  - {len(train_subset)} imágenes para Entrenamiento")
    print(f"  - {len(val_subset)} imágenes para Validación")
    print(f"  - {len(test_subset)} imágenes para Prueba")
    
    return train_loader, val_loader, test_loader