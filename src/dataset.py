import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir='./data', batch_size=64, img_size=32):
    """
    Prepara y devuelve los DataLoaders para los conjuntos de 
    entrenamiento, validación y prueba.
    """
    
    # 1. Definir las transformaciones para las imágenes
    #    - transforms.ToTensor() convierte la imagen a un tensor de PyTorch 
    #      y normaliza los valores de los píxeles a [0, 1].
    #    - transforms.Resize() ajusta el tamaño de la imagen.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # 2. Cargar el CONJUNTO DE ENTRENAMIENTO COMPLETO del disco
    full_train_dataset = datasets.FashionMNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )

    # 3. Dividir el conjunto de entrenamiento en TRAIN y VALIDATION
    #    Usaremos un 80% para entrenar y un 20% para validar.
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # random_split realiza la división de forma aleatoria
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

    # 4. Cargar el CONJUNTO DE TEST por separado
    test_dataset = datasets.FashionMNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )

    # 5. Crear los DataLoaders
    #    - shuffle=True para el conjunto de entrenamiento es crucial para que
    #      el modelo no vea los datos en el mismo orden en cada época.
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset listo:")
    print(f"  - {len(train_subset)} imágenes para Entrenamiento")
    print(f"  - {len(val_subset)} imágenes para Validación")
    print(f"  - {len(test_dataset)} imágenes para Prueba")
    
    return train_loader, val_loader, test_loader