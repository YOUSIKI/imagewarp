"""Warp images by drawing lines between corresponding points."""

import json
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from tyro.extras import SubcommandApp


class Window:
    """Window."""

    image_a_path: Path
    image_b_path: Path
    image_a: MatLike
    image_b: MatLike
    image_concat: MatLike
    point_pairs: list[tuple[tuple[int, int], tuple[int, int]]]
    point_a: tuple[int, int] | None
    point_b: tuple[int, int] | None
    output_path: Path
    pairs_path: Path

    def __init__(
        self, output_path: Path, image_a_path: Path | None, image_b_path: Path | None
    ):
        """Initialize the window."""
        # If the image paths are not provided, use the default images
        if image_a_path is None:
            image_a_path = output_path / "image_a.png"
        if image_b_path is None:
            image_b_path = output_path / "image_b.png"

        # Read images as color images
        image_a = cv2.imread(str(image_a_path), cv2.IMREAD_COLOR)
        image_b = cv2.imread(str(image_b_path), cv2.IMREAD_COLOR)

        # Resize image_b to the same height as image_a while keeping the aspect ratio
        height_a, _, _ = image_a.shape
        height_b, width_b, _ = image_b.shape
        height_b, width_b = height_a, int(width_b * height_a / height_b)
        image_b = cv2.resize(image_b, (width_b, height_b))

        self.image_a_path = image_a_path
        self.image_b_path = image_b_path
        self.image_a = image_a
        self.image_b = image_b
        self.image_concat = np.hstack((image_a, image_b))
        self.point_pairs = []
        self.point_a = None
        self.point_b = None
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.pairs_path = self.output_path / "pairs.json"

        # Save resized images
        cv2.imwrite(str(self.output_path / "image_a.png"), image_a)
        cv2.imwrite(str(self.output_path / "image_b.png"), image_b)

        # Load the existing point pairs from the output file
        if self.pairs_path.exists():
            with self.pairs_path.open("r") as f:
                data = json.load(f)
                self.point_pairs = data["point_pairs"]
                for point_a, point_b in self.point_pairs:
                    assert 0 <= point_a[0] < image_a.shape[1]
                    assert 0 <= point_a[1] < image_a.shape[0]
                    assert 0 <= point_b[0] < image_b.shape[1]
                    assert 0 <= point_b[1] < image_b.shape[0]

        # Create a window with normal mode
        cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
        # Set the mouse callback function
        cv2.setMouseCallback("Images", self.on_mouse_event)
        # Display the concatenated image
        self.refresh()

        # Wait for pressing q to quit
        while True:
            key = cv2.waitKey(0)
            if key == ord("q"):
                self.save()
                break
            elif key == ord("s"):
                self.on_new_point_pair()
            elif key == ord("u"):
                self.undo()
            elif key == ord("w"):
                self.warp()
            else:
                print(f"Unknown key: {key}")

    def __del__(self):
        """Destroy the window."""
        cv2.destroyAllWindows()

    def refresh(self):
        """Refresh the window."""
        cv2.imshow("Images", self.render())

    def render(self) -> MatLike:
        """Render the window."""
        img = self.image_concat.copy()

        # Draw lines between the corresponding points
        for point_a, point_b in self.point_pairs:
            # Shift the x-coordinate of point_b by the width of image_a
            point_b = self.shift_point_b(point_b)
            # Draw a line between the two points
            cv2.line(img, point_a, point_b, (0, 255, 0), 1)
            # Draw a small circle at each point
            cv2.circle(img, point_a, 2, (0, 255, 0), -1)

        # Draw the clicked points waiting for the corresponding point
        if self.point_a is not None:
            cv2.circle(img, self.point_a, 3, (0, 0, 255), -1)
        if self.point_b is not None:
            point_b = self.shift_point_b(self.point_b)
            cv2.circle(img, point_b, 3, (0, 0, 255), -1)

        return img

    def on_new_point_pair(self):
        """Add a new point pair."""
        if self.point_a is not None and self.point_b is not None:
            self.point_pairs.append((self.point_a, self.point_b))
            self.point_a = None
            self.point_b = None
            self.refresh()
            self.save()

    def save(self):
        """Save the point pairs to the output file."""
        with self.pairs_path.open("w") as f:
            json.dump(
                {"point_pairs": self.point_pairs},
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"Saved to {self.output_path}")

    def on_mouse_event(self, event, x, y, flags, param):
        """Handle mouse events."""
        # If the event is a left mouse button click
        # Select a point on the image
        if event == cv2.EVENT_LBUTTONDOWN:
            # If the click is on the left image
            if x < self.image_a.shape[1]:
                self.point_a = (x, y)
            # If the click is on the right image
            else:
                self.point_b = self.unshift_point_b((x, y))
            self.refresh()
        # If the event is a right mouse button click
        # Cancel a paired point nearest to the clicked point
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.cancle_point_pair(x, y)

    def cancle_point_pair(self, x, y):
        """Cancel a paired point nearest to the point."""
        min_distance = float("inf")
        nearest_point_pair = None
        for idx, (point_a, point_b) in enumerate(self.point_pairs):
            distance = min(
                np.linalg.norm(np.array([x, y]) - np.array(point_a)),
                np.linalg.norm(
                    np.array([x, y]) - np.array(self.shift_point_b(point_b)),
                ),
            )
            if distance < min_distance:
                min_distance = distance
                nearest_point_pair = idx
        if nearest_point_pair is not None:
            self.point_pairs.pop(nearest_point_pair)
            self.refresh()
            self.save()

    def undo(self):
        """Undo the last point pair."""
        if len(self.point_pairs) > 0:
            self.point_pairs.pop()
            self.refresh()
            self.save()

    def shift_point_b(self, point_b):
        """Shift the x-coordinate of point_b by the width of image_a."""
        return (point_b[0] + self.image_a.shape[1], point_b[1])

    def unshift_point_b(self, point_b):
        """Shift the x-coordinate of point_b by the width of image_a."""
        return (point_b[0] - self.image_a.shape[1], point_b[1])

    def warp(self):
        """Warp the image_b to the image_a."""
        print("Warping...")
        # Calculate a homography, and warp img to voxel
        dst_pts = np.array(
            [list(point_a) for point_a, _ in self.point_pairs], dtype=np.float32
        )
        src_pts = np.array(
            [list(point_b) for _, point_b in self.point_pairs], dtype=np.float32
        )
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        np.save(self.output_path / "homography.npy", homography)
        warped_b = cv2.warpPerspective(
            self.image_b, homography, (self.image_a.shape[1], self.image_a.shape[0])
        )
        cv2.imwrite(str(self.output_path / "warped_b.png"), warped_b)
        # Concatenate image_a and warped_b and save
        concatenated = np.hstack((self.image_a, warped_b))
        cv2.imwrite(str(self.output_path / "warped_concat.png"), concatenated)


app = SubcommandApp()


@app.command
def draw(output: Path, image_a: Path | None = None, image_b: Path | None = None):
    """Draw lines between corresponding points to warp images.

    Args:
        output (Path): the output file to save the point pairs
        image_a (Path): the first image
        image_b (Path): the second image
    """
    Window(output_path=output, image_a_path=image_a, image_b_path=image_b)


@app.command
def apply(image_a: Path, image_b: Path, homography: Path, output: Path):
    """Apply the homography to warp the second image to the first image.

    Args:
        image_a (Path): the first image
        image_b (Path): the second image
        homography (Path): the homography
        output (Path): the output file to save the warped image
    """
    image_a = cv2.imread(str(image_a), cv2.IMREAD_COLOR)
    image_b = cv2.imread(str(image_b), cv2.IMREAD_COLOR)
    # Resize image_b to the same height as image_a while keeping the aspect ratio
    height_a, _, _ = image_a.shape
    height_b, width_b, _ = image_b.shape
    height_b, width_b = height_a, int(width_b * height_a / height_b)
    image_b = cv2.resize(image_b, (width_b, height_b))
    homography = np.load(homography)
    warped_b = cv2.warpPerspective(
        image_b, homography, (image_a.shape[1], image_a.shape[0])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), warped_b)


def main():
    """Entrypoint."""
    app.cli()


if __name__ == "__main__":
    main()
