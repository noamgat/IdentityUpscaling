from torchvision.datasets import CelebA

#Workaround annoying issue:
for _ in range(20):
    try:
        dataset = CelebA(root='CelebA_Raw', split='all', download=True)
        break
    except Exception as e:
        print(e)
print("Done")