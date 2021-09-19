"""
__author__: Le Trung Phong
__email__ = letrungphong95@gmail.com
"""

class CustomTransforms:
    """
    """
    def __init__(self):
        """
        """
        pass 

    def __call__(self, sample):
        return {
            "image": sample['image'],
            "labels": sample['labels']
        }