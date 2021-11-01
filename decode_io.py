import nvidia.dali.fn as fn
data = 'video_path.mp4'

images, labels = fn.readers.video(device="gpu",
                                  file_root=data,
                                  sequence_length=8,
                                  stride=8,
                                  name="Reader")

