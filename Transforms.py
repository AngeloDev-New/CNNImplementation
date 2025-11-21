import torchvision.transforms.functional as F
import torchvision.transforms as T
import torch
class CropSides:
    """
    CropSides(top, right=None, bottom=None, left=None)
    Valores em porcentagem (0.0 a 1.0)
    """
    def __init__(self, top, right=None, bottom=None, left=None):
        # se passar apenas um valor, aplica para todos
        if right is None: right = top
        if bottom is None: bottom = top
        if left is None: left = right  # mantém simetria

        self.top_pct = top
        self.right_pct = right
        self.bottom_pct = bottom
        self.left_pct = left

    def __call__(self, img):
        #   # Caso 1 — PIL Image
        # if isinstance(img, Image.Image):
        #     w, h = img.size

        # Caso 2 — Tensor CxHxW
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape

        # Caso 3 — NumPy array HxWxC
        elif isinstance(img, np.ndarray):
            h, w = img.shape[:2]

        else:
            raise TypeError(f"Tipo de imagem não suportado: {type(img)}")


        # converter porcentagem para pixels
        top_cut = int(h * self.top_pct)
        bottom_cut = int(h * self.bottom_pct)
        left_cut = int(w * self.left_pct)
        right_cut = int(w * self.right_pct)

        # calcular dimensões finais
        new_h = h - top_cut - bottom_cut
        new_w = w - left_cut - right_cut

        # garantir que não fique negativo
        new_h = max(1, new_h)
        new_w = max(1, new_w)

        return F.crop(img,
                      top=top_cut,
                      left=left_cut,
                      height=new_h,
                      width=new_w)
