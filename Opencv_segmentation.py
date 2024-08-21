import os
import numpy as np
import cv2

# 定义路径
root_dir = '/home/yanwenli/light-weight-refinenet/test/data/'  # 更新为你的根目录
output_dir = '/home/yanwenli/light-weight-refinenet/test/out_put_image/'  # 更新为你的输出目录

# 检查输出目录是否存在，不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历所有300个场景
for i in range(1, 301):
    scene_path = os.path.join(root_dir, str(i), 'scene.npy')  # 构建每个场景的路径
    data = np.load(scene_path, allow_pickle=True).item()
    
    # 读取图像数据
    image = data.get('rgb')
    
    if image is None:
        print(f"Failed to load image {scene_path}")
        continue
    
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny边缘检测
    edged = cv2.Canny(blurred, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果找到轮廓，则处理
    if len(contours) >= 2:  # 假设场景中有两个主要物体
        # 按面积排序，找出最大的两个轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        
        def classify_object(contour):
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h  # 长宽比
            
            # 通过长宽比来判断是圆柱体还是矩形物体
            if 0.8 <= aspect_ratio <= 1.2:  # 圆柱体
                return "cylinder"
            else:  # 矩形
                return "rectangle"
        
        # 判断哪个是圆柱体，哪个是矩形物体
        if classify_object(contours[0]) == "cylinder":
            cylinder_pos = cv2.moments(contours[0])
            obstacle_pos = cv2.moments(contours[1])
        else:
            cylinder_pos = cv2.moments(contours[1])
            obstacle_pos = cv2.moments(contours[0])

        # 计算质心
        def compute_centroid(M):
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                return np.array([cX, cY])
            else:
                return np.array([0, 0])
        
        cylinder_pos = compute_centroid(cylinder_pos)
        obstacle_pos = compute_centroid(obstacle_pos)

        # 定义货架的左边缘位置
        left_shelf_pos = np.array([0, cylinder_pos[1]])  # 左货架在最左边
        right_shelf_pos= np.array([1130, cylinder_pos[1]])
        
        # 计算 d1 和 d2
        d1 = np.linalg.norm(cylinder_pos - left_shelf_pos)  # 圆柱体到左货架的距离
        d2 = np.linalg.norm(cylinder_pos - obstacle_pos)    # 圆柱体到矩形障碍物的距离
        d3 = np.linalg.norm(cylinder_pos - right_shelf_pos) # 圆柱体到右货架的距离

        d3=-d3

        # 打印或保存数据
        print(f"Scene {i}: Cylinder position: {cylinder_pos}, d1: {d1}, d2: {d2},d3,{d3}")


        # 保存每个场景的处理结果
        output_data = os.path.join(output_dir,f"data_txt{i}.txt")
        with open(output_data,'w') as f:
            f.write(f" {d1} {d2} {d3}\n")

    else:
        print(f"Could not find two objects in scene {i}")

print("All scenes processed.")
