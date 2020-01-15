# Medical-image-processing-
处理各种格式医疗图像的代码


1. 介绍

-  常见医疗图像格式: DICOM（医学数字成像和通信），NIFTI（神经影像信息技术），PAR / REC（飞利浦MRI扫描仪格式），ANALYZE（梅奥医学影像），NRRD（近原始光栅数据） ）和MNIC。

- 需要安装的包: $Pydicom$  $SimpleITK$ $PIL$ $nibabel$ $nrrd$

  ```text
  conda install pydicom --channel conda-forge
  ```

- 看医疗图像的软件： ITK SNAP, 3D slicer

2. DICOM图像处理

   Pydicom支持DICOM格式的读取：可以将dicom文件读入python结构，同时支持修改后的数据集可以再次写入DICOM格式文件。

   **读取Dicom文件并显示**

   ```python
   import pydicom
   import matplotlib.pyplot as plt
   ds = pydicom.dcmread(file)
   plt.figure(figsize=(10, 10))
   plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
   plt.show()
   ```

   **完整CT预处理示例**

   对于CT图像，通常以患者的一次拍摄为一个文件夹，文件夹下有一序列的dicom文件，每个文件称为一个切片（slice）。但是每个患者的情况不同，所以slice间的间距不同，并且可能slice的排序也不同，因此需要在训练数据前做预处理。

   CT扫描中的测量单位是Hounsfield单位（HU），默认情况下，从DICOM文件中获得的值是HU这个单位

   这里展示利用**pydicom**对一个包含多个患者数据的文件夹处理的例子

   - 导入libraries，将所有患者列出来

   ```python
   import numpy as np # linear algebra
   import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
   import dicom
   import os
   import scipy.ndimage
   import matplotlib.pyplot as plt
   
   from skimage import measure, morphology
   from mpl_toolkits.mplot3d.art3d import Poly3DCollection
   
   # 包含所有患者目录的根目录
   INPUT_FOLDER = '../input/sample_images/'
   patients = os.listdir(INPUT_FOLDER)
   patients.sort()
   ```

   - 扫描一个患者的目录，加载所有slice，按z方向排序，并获得切片厚度

   ```python
   def load_scan(path):
       slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
       slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
       try:
           slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
       except:
           slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
       for s in slices:
           s.SliceThickness = slice_thickness
       return slices
   ```

   - 提取患者信息

   ```python
   def loadFileInformation(filename):
       information = {}
       ds = dicom.read_file(filename)    
       information['PatientID'] = ds.PatientID
       information['PatientName'] = ds.PatientName
       information['PatientBirthDate'] = ds.PatientBirthDate
       information['PatientSex'] = ds.PatientSex
       information['StudyID'] = ds.StudyID
       information['StudyDate'] = ds.StudyDate
       information['StudyTime'] = ds.StudyTime
       information['InstitutionName'] = ds.InstitutionName
       information['Manufacturer'] = ds.Manufacturer
       information['NumberOfFrames'] = ds.NumberOfFrames    
       return information
   ```

   - 将HU值超出边界之外的像素置为0，再重新缩放

   ```python
   def get_pixels_hu(slices):
       image = np.stack([s.pixel_array for s in slices])
       # 转换为int16，int16是ok的，因为所有的数值都应该 <32k
       image = image.astype(np.int16)
       # 设置边界外的元素为0
       image[image == -2000] = 0
       # 转换为HU单位
       for slice_number in range(len(slices)):
           intercept = slices[slice_number].RescaleIntercept
           slope = slices[slice_number].RescaleSlope
           if slope != 1:
               image[slice_number] = slope * image[slice_number].astype(np.float64)
               image[slice_number] = image[slice_number].astype(np.int16)
           image[slice_number] += np.int16(intercept)
       return np.array(image, dtype=np.int16)
   ```

   - 查看一个患者的图像

   ```python
   first_patient = load_scan(INPUT_FOLDER + patients[0])
   first_patient_pixels = get_pixels_hu(first_patient)
   plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
   plt.xlabel("Hounsfield Units (HU)")
   plt.ylabel("Frequency")
   plt.show()
   # 显示一个中间位置的切片
   plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
   plt.show()
   ```

   - CT重新采样到[1 1 1]（由于不同的扫描，切片的距离可能不同）

   ```python
   def resample(image, scan, new_spacing=[1,1,1]):
       # Determine current pixel spacing
       spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
       resize_factor = spacing / new_spacing
       new_real_shape = image.shape * resize_factor
       new_shape = np.round(new_real_shape)
       real_resize_factor = new_shape / image.shape
       new_spacing = spacing / real_resize_factor
       image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
       return image, new_spacing
   pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
   print("Shape before resampling\t", first_patient_pixels.shape)
   print("Shape after resampling\t", pix_resampled.shape)
   ```

   - 用matplotlib画3D图像

   ```python
   def plot_3d(image, threshold=-300):
       # Position the scan upright, 
       # so the head of the patient would be at the top facing the camera
       p = image.transpose(2,1,0)
       verts, faces = measure.marching_cubes(p, threshold)
       fig = plt.figure(figsize=(10, 10))
       ax = fig.add_subplot(111, projection='3d')
       # Fancy indexing: `verts[faces]` to generate a collection of triangles
       mesh = Poly3DCollection(verts[faces], alpha=0.70)
       face_color = [0.45, 0.45, 0.75]
       mesh.set_facecolor(face_color)
       ax.add_collection3d(mesh)
       ax.set_xlim(0, p.shape[0])
       ax.set_ylim(0, p.shape[1])
       ax.set_zlim(0, p.shape[2])
       plt.show()
   
   ```

   - 数据归一化，对关注的阈值段内的数据进行归一化操作

   ```python
   MIN_BOUND = -1000.0
   MAX_BOUND = 400.0
   def normalize(image):
       image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
       image[image>1] = 1.
       image[image<0] = 0.
       return image
   
   ```

   - 将数据生成通用视频格式

   ```python
   def writeVideo(img_array):
       frame_num, width, height = img_array.shape
       filename_output = filename.split('.')[0] + '.avi'    
       video = cv2.VideoWriter(filename_output, -1, 16, (width, height))    
       for img in img_array:
           video.write(img)
       video.release()
   
   ```

   利用**SimpleITK**处理DICOM文件

   ```python
   import SimpleITK as sitk
   import numpy as np
   #读取一个序列
   reader = sitk.ImageSeriesReader()
   dicom_names = reader.GetGDCMSeriesFileNames(case_path)
   reader.SetFileNames(dicom_names)
   image = reader.Execute()
   image_array = sitk.GetArrayFromImage(image) # z, y, x 切片数，宽，高
   origin = image.GetOrigin() # x, y, z
   spacing = image.GetSpacing() # x, y, z
   #归一化
   resample = sitk.ResampleImageFilter()
   resample.SetOutputDirection(image.GetDirection())
   resample.SetOutputOrigin(image.GetOrigin())
   newspacing = [1, 1, 1]
   resample.SetOutputSpacing(newspacing)
   newimage = resample.Execute(image)
   
   #读取单张图片
   image = sitk.ReadImage(slice_path)
   image_array = sitk.GetArrayFromImage(image) # z, y, x
   
   ```

3. NIFTI图像处理

   对于nii.gz格式文件，使用**SimpleITK**

   ```python
   import SimpleITK as sitk
   import skimage.io as io
    
   def read_img(path):
       img = sitk.ReadImage(path)# path = 'F:/my_data/t1ce.nii.gz'
       data = sitk.GetArrayFromImage(img)#channel_first
       sitk.WriteImage(data,'***.nii.gz')#保存nii文件
       return data
   
   ```

   使用**Nibabel**

   ```python
   import nibabel as nib
   import matplotlib.pyplot as plt
    
   def read_data(path):
       img=nib.load(path)
       img_array = img.get_data()#channel last,存放图像数据的矩阵 
       affine_array = img.affine.copy()#get the affine array, 定义了图像数据在参考空间的位置
       img_head = img.header.copy(); #get image metadat, 图像的一些属性信息，采集设备名称，体素的大小，扫描层数
       #获取其他一些信息的方法
   		img.shape # 获得维数信息
   		img.get_data_dtype() # 获得数据类型
   		img_head.get_data_dtype() #获得头信息的数据类型
   		img_head.get_data_shape()# 获得维数信息
   		img_head.get_zooms() #获得体素大小
       return img_array,affine_array,img_head
    
   def save_data(img_array,affine_array,img_head):#保存处理后的nii文件
     new_nii = nb.Nifti1Image(img_array,affine_array,img_head)
     nb.save(new_nii,'new_test.nii.gz')
   
   ```

4. 处理PAR/REC

   转为NIFTI

   ```python
   fns = glob(os.path.join(img_dir, '*.par'))#or .PAR
   for fn in fns:
     print(f'Converting image:{fn}')
     img = nib.load(fn)
     _, base, _ = split_filename(fn)
     out_fn = os.path.join(out_dir, base+'nii.gz')
     nifti=nib.Nifti1Image(img.dataobj,img.affine, header=img.header)
     nifti.set_data_dtype('<f4')
     nifti.to_filename(out_fn)
   
   ```

5. ANALYZE格式处理

   Analyze格式储存的每组数据组包含2个文件，一个为数据文件，其扩展名为.img，包含二进制的图像资料；另外一个为头文件，扩展名为.hdr，包含图像的元数据

   nibabel可以直接读取

   ```python
   hdr = nib.load(add).get_data()
   
   ```

6. NRRD格式处理

   ```python
   nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
   #nrrd_data 保存图像的多维矩阵
   #nrrd_options 保存图像的相关信息
   >>
   {u'dimension': 3, # 维度
    u'encoding': 'raw', # 编码方式
    u'endian': 'little', # 
    u'keyvaluepairs': {},
    u'kinds': ['domain', 'domain', 'domain'], # 三个维度的类型
    u'sizes': [30, 30, 30], #三个维度的大小
    u'space': 'left-posterior-superior', # 空间信息
    u'space directions': [['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']],
    u'space origin': ['0', '0', '0'],
    u'type': 'short'}
   
   
   ```

7. MNIC格式处理

   医疗图像 NetCDF 工具包

8. 格式转换

   **dicom 转换成 NIFTI**

    dicom2nii（https://www.nitrc.org/projects/dcm2nii/）是一个用来把 DICOM 转换为 NIFTI 的工具。nibabel 是一个读写 nifiti 文件的 python 库。如果你你想把 DICOM 转换成 NIFTI，可以使用自动转换的工具（例如，dcm2nii）。python2 下的库 dcmstack 可以将一系列的 DICOM 文件堆叠成多维数组。这些数组能够被写成 NIFTI 的文件，同时还加上一个可选的头部扩展，这个头文件包含原始 DICOM 文件的所有元数据。python3 提供了一个新的用来完成这个格式转换的库--dicom2nifti。我建议读者去查看一下 nipy 这个项目。

   **DICOM 到 MINC 的转换**

   脑成像中心（BIC）的 MINC 团队开发了将 DICOM 转换为 MINC 格式的工具。这个工具是用 C 语言开发的，它在 GitHub 上的地址为：https://github.com/BIC-MNI/minc-tools/tree/master/conversion/dcm2mnc

   **NIfTI 或 ANALYZE 转换成 MINC**

   脑成像中心（BIC）的 MINC 团队还开发了将 NIfTI 或 ANALYZE 转换成 MINC 格式图像的工具。这个程序叫做 nii2mnc。包括 nii2mnc 的很多格式转换工具在这里可以看到：https://github.com/BIC-MNI/minc-tools/tree/master/conversion。

参考：

处理医疗影像的python利器：PyDicomhttps://zhuanlan.zhihu.com/p/59413289

https://www.jiqizhixin.com/articles/2017-07-31
