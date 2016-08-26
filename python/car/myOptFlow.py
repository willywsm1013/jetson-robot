import cv2
import numpy as np

if __name__ == '__main__':
    video_cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    out_video = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width,height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 10,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict( maxCorners = 500,
                           qualityLevel = 0.1,
                           minDistance = 7,
                           blockSize = 7 )
    track_len = 10
    detect_interval = 1
    tracks = []
    frame_idx = 0
    while True:
        ret, frame = video_cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        if len(tracks) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
            tracks = new_tracks
            avg_tracks = []
            for tr in tracks:
                '''[vx,vy,x,y] = cv2.fitLine(np.asarray(tr),cv2.DIST_L2,0,0.01,0.01)
                length = cv2.arcLength(np.asarray(tr), False)
                avg_tracks.append([(tr[-1][0],tr[-1][1]), (tr[-1][0]+length*vx,tr[-1][1]+length*vy)])'''
                avg_tracks.append([(tr[-1][0],tr[-1][1]), (tr[0][0],tr[0][1])])
            #cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
            cv2.polylines(vis, [np.int32(tr) for tr in avg_tracks], False, (0, 255, 0))
            cv2.putText(vis, 'track count: %d' % len(tracks), (20,50), font, 1, (255,255,255))

        if frame_idx % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])


        frame_idx += 1
        prev_gray = frame_gray
        cv2.imshow('lk_track', vis)
        out_video.write(vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    out_video.release()
    cv2.destroyAllWindows()
