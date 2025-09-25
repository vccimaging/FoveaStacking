from time import sleep
import numpy as np
import torch
import time
from .utils_io import demosaic
from .dpp_ctrl_NN import *
import torch.optim as optim

###################
# DPP class for deformable phase plate
###################
class DPP_CTL():
    def __init__(self,verbose=False,device='cpu',model_weight="deeplens/dpp_ctrl_NN/weight",N_zern=28,connect=True):
        try:
            from dpp_ctrl import api_dpp
        except:
            print("Failed to import dpp_ctrl, make sure you have installed it")
        
        self.device = device
        if connect:
            self.dppctl = api_dpp.initialize(verbose_debug_info=verbose)
            self.dppctl.connect_device(port_name="/dev/cu.usbserial-A50285BI") #"")  # "COM3" for windows experiment, "/dev/cu.usbserial-A50285BI" for Mac 
            self.load_calibration()

        print(f"DPP ctl device for NN model: {device}")
        # load the NN models
        self.N_zern = N_zern # 28 Zernike coefficients for 4th order
        self.encoder = Encoder(input_size=self.N_zern)
        self.encoder.load_state_dict(torch.load(f'{model_weight}/encoder.pt',map_location=device))
        self.decoder = Decoder(output_size=N_zern)
        self.decoder.load_state_dict(torch.load(f'{model_weight}/decoder.pt',map_location=device))

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    def load_calibration(self,calibration_file="",operation_mode='v'):
        ''' Load calibration file from the device
        
        Args:
            calibration_file ([str]): [path to the calibration file]
        '''
        success = self.dppctl.load_precalibration(calibration_file, operation_mode=operation_mode)
        if not success:
            raise Exception("Failed to load precalibration file: ", calibration_file)
        success = self.dppctl.get_ampls_limits()
        
        self.infl_matrix = torch.tensor(self.dppctl.influence_matrix).float().to(self.device) # shape [91,63]
        #round to 3 decimal places
        self.flat_field = torch.tensor(self.dppctl.flatten_field_coefficients).float().to(self.device) # shape [91]
        # self.flat_field = torch.round(self.flat_field*1000)/1000


    def apply_zern(self,zern_amp,method="NN",scale=1.0):
        ''' Apply Zernike phase to the DPP device 
        
        Args:
            zern_amp ([list]): [list of Zernike coefficients]

        Returns:
            volt_written: True if the voltage is written to the device
        '''
        volt = self.cal_volt_from_zern(torch.tensor(zern_amp).float().to(self.device),method=method,scale=scale)
        print(f"Voltage calculated: {volt/270}")
        volt = volt.detach().cpu().numpy()
        # self.apply_volt(np.zeros_like(volt))
        # time.sleep(0.1)
        volt_written, volt_converted = self.apply_volt(volt)
        return volt_written

    def apply_volt(self,volts):
        ''' Apply voltage to the DPP device
        
        Args:
            volts ([numpy]): [numpy of voltages]
        
        Returns:
            volt_written: True if the voltage is written to the device
            converted_voltages: converted voltages
        '''
        self.dppctl.volts_calculated = True
        self.dppctl.corrected_voltages = volts
        self.dppctl.voltages = volts
        self.dppctl._send_voltages()

        return self.dppctl.volts_written, self.dppctl.converted_voltages
    
    def cal_volt_from_zern(self,zern_amp,method="NN",scale=1.0):
        ''' compute the voltage applied to the DPP device from the Zernike phase, use NN model
        Args:
            zern_amp: Zernike amplitudes of shape [28] in mm, type: torch.tensor
        '''
        if method == "NN":
            print("Using NN model to compute the voltage")
            with torch.enable_grad():
                if not torch.is_tensor(zern_amp):
                    zern_amp = torch.tensor(zern_amp).float().to(self.device)
                else:
                    zern_amp = zern_amp.clone().detach().float().to(self.device)
                    
                if zern_amp.dim() == 1:
                    zern_amp = zern_amp.unsqueeze(0)
                zern_amp = zern_amp*1000 # convert to um
                zern_norm = zern_amp.norm()
                print(f"Zernike amplitude: {zern_norm}")
                # scale = 1.0 - 0.06*torch.clip(zern_norm,0,2)
                # scale = 0.9
                # zern_amp = zern_amp*scale # scaling
                volt = self.encoder(zern_amp).detach()
                volt *= 0
                volt.requires_grad = True
                optimizer = optim.Adam([volt], lr=0.01)
                criterion = zernike_MSE
                # torch.manual_seed(0)
                for i in range(10000):
                    zern = self.decoder(volt)
                    loss = criterion(zern[...,:], zern_amp[...,:])
                    if loss<0.005: 
                        break
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if i%10==0:
                        with torch.no_grad():
                            volt.clip_(0, 1)
                with torch.no_grad():
                    volt.clip_(0, 1)
                print(f"loss: {loss.item()}")
                volt = volt.squeeze().detach().cpu() * 270
            return volt*scale
        elif method == "linear":
            print("Using linear model to compute the voltage")
            try:
                from dpp_ctrl import getvolt
            except:
                print("Failed to import getvolt from dpp_ctrl, make sure you have customised it")
            zern_amp_full = torch.zeros(91).float().to(zern_amp.device)
            zern_amp = zern_amp*1000 # converting to um
            zern_norm = zern_amp.norm()
            print(f"Zernike amplitude: {zern_norm}")
            # scale = 1.0 - 0.4*torch.clip(zern_norm,0,1.4)
            zern_amp_full[:len(zern_amp)] = zern_amp # scaling
            if self.dppctl.minus_correction_flag:
                zern_amp_full -= self.flat_field
            else:
                zern_amp_full += self.flat_field
            
            infl_matrix = self.infl_matrix
            volt = getvolt.solve_infl_matrix(infl_matrix.cpu().numpy(), zern_amp_full.detach().cpu().numpy().astype(np.double), self.dppctl.max_voltage)

            # volt_square = torch.linalg.lstsq(self.infl_matrix, zern_amp_full).solution
            volt = torch.tensor(volt).float().to(zern_amp.device) # converting to torch tensor
            return volt*scale
        else:
            raise Exception("Method not implemented")

    # def _get_device_status(self):
    #        ''''
    #    Get the status of the DPP device, it will print the uint16 voltage of 64 channels of voltages
    #    '''
    #    self.dppctl._get_device_status()
    
    def disconnect(self):
        self.dppctl.close()

###################
# FTL class for focus tunable lens
###################
class FTL_CTL():
    def __init__(self,lens_idx=None) -> None:
        try:
            # from optoKummenberg import UnitType
            from optoICC import connect, DeviceModel
        except:
            print("Failed to import ICC4Control, make sure you have installed it")

        #Connecting to board. Port can be specified like connect(port='COM12')
        self.icc4c = connect()

        print("Board info")
        #Getting board info
        serial_number = self.icc4c.EEPROM.GetSerialNumber().decode('UTF-8')
        fw_version = f"{self.icc4c.Status.GetFirmwareVersionMajor()}.{self.icc4c.Status.GetFirmwareVersionMinor()}.{self.icc4c.Status.GetFirmwareVersionRevision()}"

        print(f"Board serial number: {serial_number}")
        print(f"Board firmware version: {fw_version}")

        # In order to drive the connected product, device type of the product has to be obtained (this is not needed if
        # automatic detection is allowed)
        connected_devices = [DeviceModel(self.icc4c.MiscFeatures.GetDeviceType(0)), DeviceModel(self.icc4c.MiscFeatures.GetDeviceType(1)),
                            DeviceModel(self.icc4c.MiscFeatures.GetDeviceType(2)), DeviceModel(self.icc4c.MiscFeatures.GetDeviceType(3))]
        print(f"Connected devices: {''.join('{},'.format(x.name) for x in connected_devices)}")
        print()
        print("Driving lens")
        self.connected_channel = None
        if lens_idx is not None:
            device = connected_devices[lens_idx]
            if device.value in [DeviceModel.EL_1030_C, DeviceModel.EL_1030_TC, DeviceModel.EL_1230_TC, DeviceModel.EL_1640_TC]:
                self.connected_channel = (device, lens_idx)
        
        if self.connected_channel is None:
            print("selected channel is not connected connected, trying to find first connected lens...")
            for idx, device in enumerate(connected_devices):
                if device.value in [DeviceModel.EL_1030_C, DeviceModel.EL_1030_TC, DeviceModel.EL_1230_TC, DeviceModel.EL_1640_TC]:
                    self.connected_channel = (device, idx)
                    break

        if self.connected_channel is None:
            print("No lens connected")
        else:
            # Each channel can also be accessed directly as icc4c.Channel_0 instead of icc4c.channel[0]
            self.lens = self.icc4c.channel[self.connected_channel[1]]
            lens_serial_number = self.lens.DeviceEEPROM.GetSerialNumber().decode('UTF-8')
            print(f"Lens {self.connected_channel[0].name} ({lens_serial_number}) found on channel {self.connected_channel[1]}")

            self.min_current = -self.lens.DeviceEEPROM.GetMaxNegCurrent() # scale in mA
            self.max_current = self.lens.DeviceEEPROM.GetMaxPosCurrent() # scale in mA
            print(f"Minimum current is {self.min_current} mA, maximum current is {self.max_current} mA")

            self.max_diopter = self.lens.LensCompensation.GetMaxDiopter()
            self.min_diopter = self.lens.LensCompensation.GetMinDiopter()
            print(f"Minimum diopter is {self.min_diopter} D, maximum diopter is {self.max_diopter} D")
    
    def disconnect(self):
        self.icc4c.disconnect()

    def SetCurrent(self,current):
        """ Set the current in mA
        notice Value has to be converted from mA to A
        
        Args:
            current ([float]): [current in mA]
        """
        self.lens.StaticInput.SetAsInput()
        self.lens.StaticInput.SetCurrent(current/1000)
    
    def SetFocalPower(self,dpt):
        """ Set the focal power in diopter
        
        Args:
            dpt ([float]): [focal power in diopter]
        """
        self.lens.StaticInput.SetAsInput()
        self.lens.StaticInput.SetFocalPower(dpt)

    def read_temperature(self):
        lens_temperature = self.lens.TemperatureManager.GetDeviceTemperature()
        print(f"Lens temperature: {lens_temperature}°C")
        return lens_temperature

###################
# Camera class
###################
class Cam():
    def __init__(
        self,
        ExposureTime=200000,
        trigger=True,
        black_level=[5.37062144, 5.35671067, 5.3578167],
        wb_ratio=[206, 244,185], # white balance ratio
        connect=True,
        PixelFormat="BayerGB8",
        ) -> None:
        
        try:
            from simple_pyspin import Camera
        except:
            print("Failed to import simple_pyspin, make sure you have installed it")
        
        if PixelFormat == "BayerGB8":
            self.max_val = 255.0
        elif PixelFormat == "BayerGB16":
            self.max_val = 65535.0
        elif PixelFormat == "Mono8":
            self.max_val = 255.0
        elif PixelFormat == "RGB8":
            self.max_val = 255.0

        if connect:
            self.cam = Camera()
            self.cam.init()
            self.setup_camera(ExposureTime=ExposureTime,trigger=trigger,PixelFormat=PixelFormat)

        # parameters for image processing
        self.black_level = np.array(black_level)/255.0
        wb_ratio = np.array(wb_ratio)
        self.wb_ratio = wb_ratio.max() / wb_ratio
    
    def post_process(self,img):
        ''' Process the image for deomosaicing, black subtraction and white balance. still in linear space.
        
        Args:
            img ([numpy]): [captured image]
        
        Returns:
            [numpy]: [processed image], range [0,1]
        '''
        img = img.astype(np.float32)/self.max_val
        img = self.demosaic(img,Bayer_pattern='GBRG')
        img = self.black_subtraction(img)
        img = self.white_balance(img)
        return img
    
    def demosaic(self,img,Bayer_pattern='GBRG'):
        ''' Demosaic the image
        
        Args:
            img ([numpy]): [captured image]
            Bayer_pattern (str, optional): [Bayer pattern]. Defaults to 'GBRG'.
        
        Returns:
            [numpy]: [demosaiced image]
        '''
        return demosaic(img,Bayer_pattern=Bayer_pattern)

    def white_balance(self,img):
        ''' White balance the image
        
        Args:
            img ([numpy]): [captured image]
        
        Returns:
            [numpy]: [white balanced image]
        '''
        assert img.max() <= 1, "Image should be in range [0,1]"
        return np.clip(img * self.wb_ratio,0,1)
    
    def black_subtraction(self,img,black_level=None):
        ''' Black subtraction
        
        Args:
            img ([numpy]): [captured image]
        
        Returns:
            [numpy]: [black subtracted image]
        '''
        assert img.max() <= 1, "Image should be in range [0,1]"
        return np.clip((img - self.black_level), 0, 1)

    def setup_camera(self, PixelFormat="BayerGB16",  ExposureTime=15000, trigger=True):
        # AcquisitionFrameRate=2,
        # Set the area of interest (AOI) to the middle half.
        self.cam.OffsetX = 0 #self.cam.SensorWidth // 4
        self.cam.OffsetY = 0 # self.cam.SensorHeight // 4
        self.cam.Width = self.cam.SensorWidth # // 2
        self.cam.Height = self.cam.SensorHeight #// 2

        # # Set the area of interest (AOI) to the middle half.
        # self.cam.OffsetX = self.cam.SensorWidth // 4
        # self.cam.OffsetY = self.cam.SensorHeight // 4
        # self.cam.Width = self.cam.SensorWidth // 2
        # self.cam.Height = self.cam.SensorHeight // 2
        
        # If this is a color camera, get the image in RGB format.
        self.cam.PixelFormat = PixelFormat
        self.cam.AcquisitionMode = 'Continuous'

        self.cam.TriggerMode = 'Off' #'Off'
        self.cam.TriggerSource = 'Software'
        self.cam.TriggerActivation = 'RisingEdge'
        self.cam.TriggerDelay = 0
        # self.cam.TriggerOverlap = 'ReadOut' # 'Off'
        if trigger:
            self.cam.TriggerMode = 'On' 
            self.trigger_state = True
        else:
            self.cam.TriggerMode = 'Off' 
            self.trigger_state = False
            # To change the frame rate, we need to enable manual control
            self.cam.AcquisitionFrameRateEnabled = False
            # self.cam.AcquisitionFrameRateAuto = 'Continuous' #Off
            # AcquisitionFrameRate = max( min( int( 1e6/(ExposureTime) - 1 ), 40), 1)
            # print(f"setting acqusition frame rate: {AcquisitionFrameRate}")
            # self.cam.AcquisitionFrameRate = 90

       # To control the exposure settings, we need to turn off auto
        self.cam.GainAuto = 'Off'
        self.cam.Gain = 0
        self.cam.ExposureAuto = 'Off'
        self.cam.ExposureTime = ExposureTime # microseconds

        # black level
        # self.cam.BlackLevelSelector = 'All'
        # self.cam.BlackLevelAuto = 'off'
        self.cam.BlackLevel = 0

        # control White balance
        # self.cam.BalanceWhiteAuto = 'Off'
        if PixelFormat != "RGB8":
            self.cam.BalanceRatioSelector = 'Red'
            self.cam.BalanceRatio = 1 #1.22
            self.cam.BalanceRatioSelector = 'Blue'
            self.cam.BalanceRatio = 1 # 1.58 # 1.3

        # If we want an easily viewable image, turn on gamma correction.
        # NOTE: for scientific image processing, you probably want to
        #    _disable_ gamma correction!
        try:
            self.cam.GammaEnabled = True
            self.cam.Gamma = 1
            if PixelFormat == "RGB8":
                self.cam.Gamma = 2.2
        except:
            print("Failed to change Gamma correction (not avaiable on some cameras).")
        
        self.cam.start()

    # def trigger(self):
    #     assert self.cam.TriggerMode == 'On', "Trigger mode is not on"
    #     self.cam.TriggerSoftware()


    def get_latch_time(self):
        self.cam.TimestampLatch()
        time_stamp = self.cam.Timestamp*1e-9
        return time_stamp

        # import PySpin
        # time_props = [x for x in self.cam.camera_attributes.keys() if 'time' in x.lower()]
        # readable_time_props = [x for x in time_props if PySpin.IsReadable(self.cam.camera_attributes[x])]
        # for props in readable_time_props:
        #    print(f"{props}: {self.cam.camera_attributes[props].GetValue()}")

    def get_frame(self):
        '''
            Get a single frame from the camera
        
        Returns:
            img: [numpy array]: [captured image]
            img_chunk: [dict]: [chunk data]
        '''
        start_time = time.time()
        latch_time = self.get_latch_time()
        if self.trigger_state:
            self.cam.TriggerSoftware() # send software trigger
        trigger_time = time.time()
        img, img_chunk = self.cam.get_array(get_chunk=True) # get chunk data by default

        cur_time = (float)(img_chunk['Timestamp'])
        post_latch_time = self.get_latch_time()
        print(f'transfer_time: {post_latch_time-cur_time}')

        # check if the image is captured after the trigger
        positive_time = cur_time-latch_time
        print(f'frame_end - trigger: {cur_time-latch_time}')
        assert positive_time>0, "Error: frame captured before triggering"
    
        get_array_time = time.time()  
        print(f"trigger time: {trigger_time-start_time}, get_array_time: {get_array_time-trigger_time}")

        return img, img_chunk

    def get_steady_frame(self):
        img_old,chunk_data = self.get_frame()
        threshold = 550
        diff_mean = threshold + 100
        while (diff_mean>threshold):
            img_new,chunk_data = self.get_frame()
            diff = img_new.astype(np.float32) - img_old.astype(np.float32)
            diff_mean = np.mean(np.abs(diff))
            img_old = img_new.copy()
            if diff_mean>threshold:
                print(f"diff_mean: {diff_mean} is larger than {threshold}, capturing again ...")
            else:
                print(f"diff_mean: {diff_mean} is smaller than {threshold}, capturing done")
        return img_new, chunk_data
    
    def __del__(self):
        pass
        # self.cam.stop()
        # self.cam.close()



if __name__ == "__main__":
    from PIL import Image
    import os
    output_dir = "results/capture_img"
    ftl_ctl = FTL_CTL(0)
    dpp_ctl = DPP_CTL(verbose=False)
    cam = Cam()
    dpt = 0.00705 * 1000 * 0.29
    ftl_ctl.SetFocalPower(dpt)

    print(f"Setting static current")

    # for dpt in np.arange(int(ftl_ctl.min_diopter), int(ftl_ctl.max_diopter)+1, int((ftl_ctl.max_diopter-ftl_ctl.min_diopter)/5)):
    #     print(f"focal power {dpt} dpt")
    #     ftl_ctl.SetFocalPower(dpt)
    #     img = cam.get_frame()
    #     Image.fromarray(img).save(os.path.join(output_dir, f'{dpt:04d}.tiff'))
    #     sleep(1)


    # dpp_ctl.dppctl.launch_test_loop()

    for zern_val in np.arange(-3,3,0.5):
        zern_list = [0,0,zern_val,0,0,0,0,0,0,0]
        print(f"Sending Zernike phase: {zern_list}")
        zern_param = dpp_ctl.apply_zern(zern_list)
        print(f"Received Zernike phase on Device: {zern_param}")
        sleep(1)

    print("Setting static current to 0 A")