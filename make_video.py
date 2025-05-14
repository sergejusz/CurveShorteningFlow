import sys
import os
import cv2

def generate_video(path, video_name, frame_rate):

    images = [img for img in os.listdir(path) if img.endswith((".png"))]
    print("Files and directories in '", path, "' :")
    images.sort()
    # prints all files
    if len(images) == 0:
        print("No images found in '", path, "'")
        return False

    print(images)
    print(len(images), " images found")

    # filename should be named chronologically for example image_0001.png, image_0002.png, ...
    # so image_0001.png should correspond to the first frame

    # Set frame from the first image
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape

    # Video writer to create .avi file
    video_path = os.path.join(path, video_name)
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, (width, height))

    # Appending images to video
    for image in images:
        video.write(cv2.imread(os.path.join(path, image)))

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    return True

   
def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Invalid command line: make_video.py image_folder video_name [frame_rate]")
        print("Example: make_video.py c:\\myimages 20")
        return
    
    image_folder = args[0]
    video_name = args[1]
    # default frame rate is 1 frame/second
    frame_rate = 1
    if len(args) >= 3: frame_rate = int(args[2])
    
    if generate_video(image_folder, video_name, frame_rate):
        print("Video generated successfully!")
    else:
        print("Video generation failed!")

if __name__ == "__main__":
    main()