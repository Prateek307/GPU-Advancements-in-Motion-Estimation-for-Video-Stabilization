import pyopencl as cl


def print_platforms(platforms):
    print("-"*60)
    print("Available Platforms:")
    for i, platform in enumerate(platforms):
        print(f"{i+1}. {platform.name}")
    print("-"*60)

def get_device_info(device):
    print("-"*60)   
    print(f'Version: {device.version}')
    print(f'Type: {cl.device_type.to_string(device.type)}')
    print(f'Extensions: {str(device.extensions.strip().split(" "))}')
    print(f'Memory (global) : {str(device.global_mem_size)}')
    print(f'Address bits: {str(device.address_bits)}')
    print(f'Max work item dims: {str(device.max_work_item_dimensions)}')
    print(f'Max work group size: {str(device.max_work_group_size)}')
    print(f'Max compute units: {str(device.max_compute_units)}')
    print(f'Driver version: {device.driver_version}')
    print(f'Image support: {str(bool(device.image_support))}')
    print(f'Deivce available: {str(bool(device.available))}')
    print(f'Compiler available: {str(bool(device.compiler_available))}')
    print("-"*60)
