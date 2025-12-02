import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split 

# --- 1. CLASE PARA COMBINAR IMAGEN Y R* ---
class ConvergenceDataset(Dataset):
    def __init__(self, original_subset, R_matrix):
        """
        original_subset: Subset original de PyTorch (contiene la imagen GT y la etiqueta).
        R_matrix: Array de NumPy con la matriz R* [Num_Muestras, Max_Iters].
        """
        self.original_subset = original_subset
        self.R_tensor = torch.from_numpy(R_matrix).float()
        
    def __len__(self):
        return len(self.original_subset)

    def __getitem__(self, idx):
        # La tupla original: (img_gt, label)
        img_gt, _ = self.original_subset[idx]  # Ignoramos la etiqueta
        R_vector_gt = self.R_tensor[idx]      # Vector R* local
        
        return img_gt, R_vector_gt
# ----------------------------------------------------


def _get_divided_indices(full_dataset, TOTAL_SAMPLES):
    """
    Función auxiliar para centralizar la selección y división estratificada de índices.
    
    Retorna:
    - selected_indices (Globales, para crear los Subsets de imágenes)
    - local_train_indices, local_val_indices, local_test_indices (Locales, para cortar R*)
    """
    # 1. Selección de Índices Globales y Targets (CRUCIAL: seed 42)
    np.random.seed(42)
    indices_pool = np.arange(len(full_dataset))
    selected_indices = np.random.choice(indices_pool, TOTAL_SAMPLES, replace=False)
    
    subset_targets = []
    for idx in selected_indices:
        subset_targets.append(full_dataset[idx][1])
    subset_targets = np.array(subset_targets)

    # 2. Creación de Índices LOCALES (0 a 9999) para la División
    local_indices_to_split = np.arange(TOTAL_SAMPLES)
    
    # 3. DIVISIÓN ESTRATIFICADA (usando índices locales)
    
    # A. Dividir primero en Train (70%) y Resto (30%)
    local_train_indices, remaining_indices_local, _, remaining_targets = train_test_split(
        local_indices_to_split,  # <-- USAMOS ÍNDICES LOCALES PARA LA DIVISIÓN
        subset_targets,
        test_size=0.3, 
        random_state=42, 
        stratify=subset_targets 
    )

    # B. Dividir el Resto
    local_val_indices, local_test_indices, _, _ = train_test_split(
        remaining_indices_local,
        remaining_targets,
        test_size=0.5, 
        random_state=42,
        stratify=remaining_targets 
    )
    
    return selected_indices, local_train_indices, local_val_indices, local_test_indices, full_dataset


def get_dataloaders(data_dir='./data', batch_size=64, img_size=32):
    """
    Prepara y devuelve los DataLoaders para TRAIN, VAL y TEST.
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

    # 3. Obtener índices locales y globales de la función auxiliar
    (selected_indices, local_train_indices, local_val_indices, local_test_indices, full_dataset) = \
        _get_divided_indices(full_dataset, TOTAL_SAMPLES)

    # 4. Crear los Subsets de PyTorch (¡USANDO ÍNDICES GLOBALES PARA REFERENCIAR EL DATASET COMPLETO!)
    
    # Los local_indices deben mapearse a selected_indices antes de usarse en Subset
    train_indices_global = selected_indices[local_train_indices]
    val_indices_global = selected_indices[local_val_indices]
    test_indices_global = selected_indices[local_test_indices]
    
    train_subset = Subset(full_dataset, train_indices_global)
    val_subset = Subset(full_dataset, val_indices_global)
    test_subset = Subset(full_dataset, test_indices_global)

    # 5. Creación de DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print(f"Dataset LISTO (Total: {TOTAL_SAMPLES} muestras, ESTRATIFICADO):")
    print(f"  - {len(train_subset)} imágenes para Entrenamiento")
    print(f"  - {len(val_subset)} imágenes para Validación")
    print(f"  - {len(test_subset)} imágenes para Prueba")
    
    return train_loader, val_loader, test_loader


def get_convergence_dataloaders(config, R_star_path='metricas/FISTA_optimal_convergence_R_star.npy'):
    """
    Carga el dataset base y lo combina con la matriz R* de tasas de convergencia.
    Retorna DataLoaders donde cada iteración devuelve (img_gt, R_vector_gt).
    """
    TOTAL_SAMPLES = 10000
    batch_size = config['data']['batch_size']
    img_size = config['data']['img_size']
    data_dir = './data'
    
    # 1. Cargar y dividir los índices (Usamos la función auxiliar centralizada)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    full_train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset_full = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([full_train_dataset, test_dataset_full])
    
    (selected_indices, local_train_indices, local_val_indices, local_test_indices, full_dataset) = \
        _get_divided_indices(full_dataset, TOTAL_SAMPLES)
        
    # Crear Subsets de Imágenes (Usando índices globales)
    train_indices_global = selected_indices[local_train_indices]
    val_indices_global = selected_indices[local_val_indices]
    test_indices_global = selected_indices[local_test_indices]
    
    train_subset = Subset(full_dataset, train_indices_global)
    val_subset = Subset(full_dataset, val_indices_global)
    test_subset = Subset(full_dataset, test_indices_global)
    
    # 2. CARGAR y DIVIDIR la Matriz R* (Usando Índices LOCALES para el corte)
    try:
        R_star_full = np.load(R_star_path) 
    except FileNotFoundError:
        print(f"ERROR: No se encontró la matriz R* en {R_star_path}.")
        print("Asegúrate de ejecutar 'python generate_fista_rates.py' primero.")
        raise
        
    # Usar los índices LOCALES para cortar la matriz R*
    R_train = R_star_full[local_train_indices]  # <-- ¡CORRECCIÓN DEL INDEXERROR!
    R_val = R_star_full[local_val_indices]      # <-- ¡CORRECCIÓN DEL INDEXERROR!
    R_test = R_star_full[local_test_indices]    # <-- ¡CORRECCIÓN DEL INDEXERROR!

    # 3. Crear los DataLoaders de Convergencia
    train_loader = DataLoader(
        ConvergenceDataset(train_subset, R_train), 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        ConvergenceDataset(val_subset, R_val), 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        ConvergenceDataset(test_subset, R_test), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print("\n✅ DataLoaders de Convergencia LISTOS (GT: Imagen + Vector R*).")

    # ... (Cerca de la línea final)

    print("\n✅ DataLoaders de Convergencia LISTOS (GT: Imagen + Vector R*).")

    # --- INFORMACIÓN DETALLADA DEL DATALOADER ---
    
    try:
        # Intentar obtener un batch de ejemplo para ver los shapes
        sample_batch = next(iter(train_loader))
        img_shape = sample_batch[0].shape
        r_vector_shape = sample_batch[1].shape
    except Exception as e:
        img_shape = "No disponible"
        r_vector_shape = "No disponible"
        # print(f"Advertencia: No se pudo obtener el shape del sample. {e}")

    print("-" * 55)
    print("DETALLES DEL DATASET DE CONVERGENCIA:")
    print("-" * 55)
    print(f"Total de Muestras (R*): {TOTAL_SAMPLES}")
    print(f"Batch Size: {batch_size}")
    print(f"Imágenes (GT) Shape por Batch: {img_shape}")
    print(f"Vector R* Shape por Batch: {r_vector_shape}")
    print("-" * 55)
    print("Contenido de cada tupla (Batch):")
    print("  [0] - Imagen GT (x*): [Batch, Canal, Alto, Ancho]")
    print("  [1] - Vector R* GT: [Batch, Max_Iters]")
    print("-" * 55)
    
    # --- FIN DE INFORMACIÓN DETALLADA ---
    return train_loader, val_loader, test_loader