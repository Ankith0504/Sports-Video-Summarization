from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
import json
with open('features.json') as labels_json:
    annotations = json.load(labels_json)
def called():
    lst_avail=[]
    for i  in annotations["labels"]:
        if i["label"] not in lst_avail:
            lst_avail.append(i["label"])
    #print(lst_avail)
    return lst_avail
def required():
    lst_avail=called()
    labels={'1':'Kick-off', '2':'Ball out of play', '3':'Clearance', '4':'Goal', '5':'Shots on target', '6':'Corner', '7':'Foul', '8':'Throw-in', '9':'Shots off target', '10':'Indirect free-kick', '11':'Yellow card', '12':'Substitution', '13':'Offside', '14':'Direct free-kick'}
    list=['1:Kick-off', '2:Ball out of play', '3:Clearance', '4:Goal', '5:Shots on target', '6:Corner', '7:Foul', '8:Throw-in', '9:Shots off target', '10:Indirect free-kick', '11:Yellow card', '12:Substitution', '13:Offside', '14:Direct free-kick']
    for i in list:
        if i.split(":")[1].strip() in lst_avail:
            print(i)
    out=[]
    req_labels=input('Enter the Highlights Required as comma Seperated Numbers:')
    inputs=req_labels.split(",")
    for i in inputs:
        out.append(labels[i])
    return out
def merge_intervals(intervals):
    if not intervals:
        return []

    merged_intervals = [intervals[0]]

    for i in range(1, len(intervals)):
        current_interval = intervals[i]
        previous_interval = merged_intervals[-1]

        if current_interval[0] <= previous_interval[1] + 1:
            # Merge the current interval with the previous one
            previous_interval = (previous_interval[0], max(current_interval[1], previous_interval[1]))
            merged_intervals[-1] = previous_interval
        else:
            # No overlap, add the current interval to the list
            merged_intervals.append(current_interval)
    return merged_intervals
def time_stamps_returned(annotations,half,req_labels):
    labels=annotations["labels"]
    time_stamps=[]
    video=""
    for i in labels:
        time_sec=i["gameTime"]
        if i["half"]==half:
            label=i["label"]
            if label in req_labels:
                if label=="Kick-off" and time_sec==0:
                    start=time_sec
                    end=start+8
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="y-card" or label=="Yellow card":
                    start=time_sec-3
                    end=start+8
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="Foul":
                    start=time_sec-2
                    end=start+7
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="Corner":
                    start=time_sec-2
                    end=start+7
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="Goal":
                    start=time_sec-3
                    end=start+10
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="Throw-in":
                    start=time_sec-3
                    end=start+7
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="Indirect free-kick":
                    start=time_sec-4
                    end=start+10
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="Shots on target":
                    start=time_sec-2
                    end=start+6
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="Shots off target":
                    start=time_sec-2
                    end=start+6
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="Ball out of play":
                    start=time_sec-2
                    end=start+6
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="Clearance":
                    start=time_sec-2
                    end=start+6
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="Direct free-kick":
                    start=time_sec-2
                    end=start+6
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="Substitution":
                    start=time_sec-2
                    end=start+6
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
                elif label=="Offside":
                    start=time_sec-2
                    end=start+6
                    time_stamp=(start,end)
                    time_stamps.append(time_stamp)
    time_stamps.sort()
    return merge_intervals(time_stamps)

# Specify the input video file, output file, and time ranges
#labels={1:"Kick-off",2:"Yellow Card",3:"Foul",4:"Corner",5:"Goal",6:"Throw-in",7:"Indirect free-kick",8:"Shots on target",9:"Shots off target",10:"Ball out of play"}
labels=required()
input_video = "1_720p.mkv"
output_video = "720p_summarized.mkv"
video_length = int(VideoFileClip(input_video).duration)
#print(video_length)
time_ranges = time_stamps_returned(annotations,input_video.strip("_")[0],labels)
print()
print("1st Half")
print(time_ranges)
# Cut and merge the video
video = VideoFileClip(input_video)
subclips = []

if len(time_ranges)!=0:
    if time_ranges[-1][1]>video_length:
        del time_ranges[-1]
    print()
    print("Extracting Video Bits of First Half:")

for start, end in time_ranges:
    print(start,end)
    subclip = video.subclip(start, end)
    subclips.append(subclip)

input_video = "2_720p.mkv"
time_ranges_2 = time_stamps_returned(annotations,input_video.strip("_")[0],labels)
video_length = VideoFileClip(input_video).duration
#print(video_length)
print()
print("2nd Half")
print(time_ranges_2)
if len(time_ranges_2)!=0:
    if time_ranges_2[-1][1]>video_length:
        del time_ranges_2[-1]
    print()
    print("Extracting Video Bits of Second Half:")
# Cut and merge the video
video = VideoFileClip(input_video)

for start, end in time_ranges_2:
    print(start,end)
    subclip = video.subclip(start, end)
    subclips.append(subclip)
print()
final_clip = concatenate_videoclips(subclips)

# Write the final video to the output file
final_clip.write_videofile(output_video, codec="libx264")
