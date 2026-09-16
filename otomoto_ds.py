import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from otomoto_scrapper import get_offer_urls_from_page, scrape_otomoto_offer


class OtomotoChunkedDataset(Dataset):

    def __init__(
        self,
        items: int,
        chunk_size: int = 200,
        start_page: int = 1,
        max_img_per_offer: int = 20,
        pages_per_fetch: int = 2,
    ):
        super().__init__()

        self.transforms = v2.Compose([
            v2.Resize(
                (224, 224),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

        self.max_img_per_offer = max_img_per_offer
        self.total_required_items = items
        self.chunk_size = chunk_size

        self.pages_per_fetch = pages_per_fetch
        self.current_page = start_page

        self.buffer = []
        self.buffer_start_idx = 0

        self.pending_urls = []
        self.current_page = start_page
        self.buffer = []
        self.buffer_start_idx = 0

        self._load_next_chunk()

    def _load_next_chunk(self):
        self.buffer_start_idx += len(self.buffer) if self.buffer else 0
        self.buffer = []

        while (
                len(self.buffer) < self.chunk_size
                and (self.buffer_start_idx + len(self.buffer)) < self.total_required_items
        ):
            if not self.pending_urls:
                for i in range(self.pages_per_fetch):
                    new_urls = get_offer_urls_from_page(self.current_page + i)
                    self.pending_urls.extend(new_urls)

                self.current_page += self.pages_per_fetch

                if not self.pending_urls:
                    break

            while self.pending_urls and len(self.buffer) < self.chunk_size:
                if (self.buffer_start_idx + len(self.buffer)) >= self.total_required_items:
                    break

                url = self.pending_urls.pop(0)

                try:
                    imgs, text = scrape_otomoto_offer(
                        url, max_photos=self.max_img_per_offer
                    )

                    if not imgs:
                        continue

                    tensor = torch.stack([self.transforms(img) for img in imgs])
                    self.buffer.append((tensor, len(imgs), text, url))

                except Exception:
                    continue


    def __len__(self):
        return self.total_required_items


    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_required_items:
            raise IndexError(f"index is out of range [0, {self.total_required_items}).")

        local_idx = idx - self.buffer_start_idx

        if local_idx < 0:
            raise IndexError(f"index {idx} is already garbage collected.")

        while local_idx >= len(self.buffer):
            self._load_next_chunk()

            local_idx = idx - self.buffer_start_idx

        return self.buffer[local_idx]