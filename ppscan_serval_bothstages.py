import thorlabs_apt as apt
import os, sys
from datetime import datetime
from time import sleep, mktime, strftime
print(os.getcwd())
# parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
parentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parentdir)
print(os.getcwd())
print(parentdir)
import Equipment.ESP_v2 as esp
import Equipment.OceanOptics as oo
import pathlib
import requests
import json
import time
import numpy as np

from experiment_logger_google import ExperimentLogger
logger = ExperimentLogger(channel="#exp_details")

def get_request(url, expected_status=200):
    response = requests.get(url=url)
    if response.status_code != expected_status:
        raise Exception("Failed GET request: {}, response: {} {}".format(url, response.status_code, response.text))

    return response


def put_request(url, data, expected_status=200):
    response = requests.put(url=url, data=data)
    if response.status_code != expected_status:
        raise Exception("Failed PUT request: {}, response: {} {}".format(url, response.status_code, response.text))

    return response


def check_connection(serverurl):
    get_request(url=serverurl, expected_status=200)


def get_dashboard(serverurl):
    response = get_request(url=serverurl + '/dashboard')
    data = response.text
    dashboard = json.loads(data)
    return dashboard


def init_cam(serverurl, bpc_file, dacs_file):
    response = get_request(url=serverurl + '/config/load?format=pixelconfig&file=' + bpc_file)
    data = response.text
    
    response = get_request(url=serverurl + '/config/load?format=dacs&file=' + dacs_file)
    data = response.text


def init_acquisition(serverurl, detector_config, ntriggers=1000000, trigger_period=1.0, exposure_time=1.0):
    # Sets the number of triggers.
    detector_config["nTriggers"] = ntriggers

    # Set the trigger mode to be software-defined.
    detector_config["TriggerMode"] = "CONTINUOUS"

    # Sets the trigger period (time between triggers) in seconds.
    detector_config["TriggerPeriod"] = trigger_period

    # Sets the sure time (time the shutter remains open) in seconds.
    detector_config["ExposureTime"] = exposure_time

    # Upload the Detector Configuration defined above
    response = put_request(url=serverurl + '/detector/config', data=json.dumps(detector_config))
    data = response.text

def acquisition_test(serverurl):

    # Starting acquisition process
    response = get_request(url=serverurl + '/measurement/start')
    data = response.text
    # print('Response of acquisition start: ' + data)

    # Example of measurement interruption
    taking_data = True
    while taking_data:
        dashboard = json.loads(get_request(url=serverurl + '/dashboard').text)
        # Stop measurement once Serval has collected all frames
        if dashboard["Measurement"]["Status"] == "DA_IDLE":
            taking_data = False

def do_collect(serverurl, data_folder, param_num, stupid):

    # Example of destination configuration (Python dictionary) for the data output
    destination = {
        "Raw": [{
            # URI to a folder where to place the raw files.
            "Base": f'file:///{data_folder}',
            # "Base": f'file:///{data_folder}/{param_num:03}',
            # "Base": pathlib.Path(os.path.join(data_folder, f'{param_num:03}')).as_uri(),
            # How to name the files for the various frames.
            "FilePattern": f'file{stupid:06}_param_{param_num:03}_',
            # "FilePattern": "raw%HHmmss_",
	    "SplitStrategy" : "SINGLE_FILE"
        }]
    }

    # Setting destination for the data output
    response = put_request(url=serverurl + '/server/destination', data=json.dumps(destination))
    data = response.text
    # print('Response of uploading the Destination Configuration to SERVAL : ' + data)

    # Getting destination for the data output from SERVAL
    response = get_request(url=serverurl + '/server/destination')
    data = response.text
    # print('Selected destination : ' + data)

    # Running acquisition process
    # acquisition_test(serverurl)
    response = get_request(url=serverurl + '/measurement/start')

    # print('Ready.')

def setup_all(serverurl):
    # Retrieve the dashboard from SERVAL
    dashboard = get_dashboard(serverurl)
    bpcFile = '/data/Serval/collect_data/TpxConfig_Pixel_20240812.bpc'
    dacsFile = '/data/Serval/collect_data/TpxConfig_Pixel_20240812.dacs'

    # Detector initialization with bpc and DACs
    init_cam(serverurl, bpcFile, dacsFile)

    # Example of getting the detector configuration from the server in JSON format
    response = get_request(url=serverurl + '/detector/config')
    data = response.text

    # Converting detector configuration data from JSON to Python dictionary and modifying values
    detectorConfig = json.loads(data)

    # Setting the timing for the acquisition     
    init_acquisition(serverurl, detectorConfig)

    # Getting the updated detector configuration  from SERVAL
    response = get_request(url=serverurl + '/detector/config')
    data = response.text

class spectrometer():
    def __init__(self, spectrometer_index = 0):
        spectrometers_object = oo.spectrometers() 
        self.devices = spectrometers_object.devices
        self.spectrometer_index = spectrometer_index

    def record_spectrum(self, data_folder, unix_timestamp,  integration_time, averages):
        spectrum = oo.spectrum(self.devices, device_ID = self.spectrometer_index, integration_time = integration_time, averages = averages)
        spectrum.collect_intensities()
        spectrum.save(data_folder, save_name = datetime.fromtimestamp(unix_timestamp).strftime('%Y%m%d_%H%M%S'))


def check_start_positions(COM_port):
    try:
        ESP_controller = esp.ESP_stage(COM_port)
        position1 = round(ESP_controller.position(1),4)
        position2 = round(ESP_controller.position(2),4)
        position3 = round(ESP_controller.position(3),4)
        positions = [position1,position2,position3]
        return positions
    
    except Exception as e:
        print(e)
        return None
    
def get_last_nonzero_pressure():
    today_str = datetime.today().strftime("%Y%m%d")
    filepath = f"C:\\Varian247HPMonitorData\\PressureMonitor\\{today_str}.txt"
    pressurelog = np.loadtxt(filepath)

    col = pressurelog[:, 5]
    nonzero_values = col[col != 0]
    if nonzero_values.size == 0:
        return None
    
    return nonzero_values[-1]

def parameter_scanny(data_folder, log_folder, delay_actuator_ID, rotation_COM_port, rotation_axis, parameters_all, spec_integration_time, spec_averages):

    # Define server url.
    # serverurl = 'http://localhost:8080'
    # serverurl =  'http://10.34.170.149:8080'
    serverurl =  'http://10.34.160.81:8080'

    # Check connection with SERVAL
    check_connection(serverurl)
    print('serval connection good')

    # setup camera config
    setup_all(serverurl)
    print('camera is set up')

    delay_motor = apt.Motor(delay_actuator_ID)
    ESP_controller = esp.ESP_stage(rotation_COM_port)
    # velocity = 1
    # ESP_controller.set_velocity(rotation_axis, velocity, chatty = True)

    parameters_num = len(parameters_all)
    parameters_filename = log_folder + 'parameter_log.txt'

    spec = spectrometer(0)
    spectra_foldernames = []
    for parameter_num in range(parameters_num):
        spectra_foldernames.append(os.path.join(log_folder,'parameter' + str(parameter_num).zfill(2)) + '/')
        if not os.path.exists(spectra_foldernames[parameter_num]):
            os.makedirs(spectra_foldernames[parameter_num])

    try:
        stupid = 0
        t0 = time.time()
        while True:
            for parameter_num in range(parameters_num):

                parameters = parameters_all[parameter_num] # [delay stage position, rotation stage position, wait time]

                ## Current time
                current_time = datetime.now()
                current_unix_time = int(mktime(current_time.timetuple()))
                # print("Current time:         ", current_time)
                # print("Current UNIX time:    ", current_unix_time)

                # stop collecting data

                t1 = time.time()
                if stupid > 0:
                    response = get_request(url=serverurl + '/measurement/stop')
                    print(f'stopped collection for paramenter {(parameter_num-1) % parameters_num} after {t1-t0:.2f} seconds')
                stupid += 1
                t2 = time.time()

                ## Desired parameter
                print("\nMoving to parameter:   " + str(parameter_num).zfill(2))
                
                ## Move to new delay stage position
                # start_delay_position_rounded = 0.
                start_delay_position_rounded = round(delay_motor.position,4)

                if start_delay_position_rounded != parameters[0]:
                    print('Delay moving to:       ' + str(parameters[0]))
                    delay_motor.move_to(parameters[0],'True')

                ## Record actual delay stage position
                # actual_delay_position = parameters[0]
                actual_delay_position = delay_motor.position
                print('Delay position:        ' + str(actual_delay_position))

                # Move to new rotation stage position
                # start_rotation_position_rounded = 0
                print(ESP_controller.position(rotation_axis))
                start_rotation_position_rounded = round(ESP_controller.position(rotation_axis),1)
                tracky = 0
                while (start_rotation_position_rounded != parameters[1]) and tracky < 3:
                    if tracky > 0:
                        print('Trying again :(')
                    print('Rotation moving to:    ' + str(parameters[1]))
                    ESP_controller.move_to(rotation_axis, parameters[1])
                    tracky += 1

                # Record actual rotation stage position
                # actual_rotation_position = parameters[1]
                actual_rotation_position = ESP_controller.position(rotation_axis)
                print('Rotation position:     ' + str(actual_rotation_position))

                # start taking data
                t3 = time.time()
                print(f'\nstarting collection for paramenter {parameter_num} after pausing for {t3-t2:.4f} seconds')
                t0 = time.time()
                do_collect(serverurl, data_folder, parameter_num, stupid)


                ## Output to file
                record_values_tpx(parameters_filename, current_unix_time, stupid, parameter_num, parameters[0], actual_delay_position, parameters[1], actual_rotation_position)
                # print('')

                ## Save spectrum
                spec.record_spectrum(spectra_foldernames[parameter_num], current_unix_time,  spec_integration_time, spec_averages)

                ## Sleep until next movement
                sleep(parameters[2]/5)
                # sleep(parameters[2] - self.spec_integration_time*self.spec_averages/1000000)


    except KeyboardInterrupt:
        print('   Data collection interrupted, stopping now')
        response = get_request(url=serverurl + '/measurement/stop')
        print(response.text)


def record_values_tpx(file, timestamp, stupid, parameter_index, delay_position, actual_delay_position, rotation_position, rotation_actual_position):
    with open(file, 'a+') as f:
        f.write('{}, {}, {}, {:.7f}, {:.7f}, {}, {}\n'.format(timestamp, stupid, parameter_index, delay_position, actual_delay_position, rotation_position, rotation_actual_position))
        sleep(0.1)

def main():
    # data folder
    filename = 'Test_1_Pressure_OnlyIon'
    foldername = 'Tpx_20250508'

    note = '800 only, looking for evidence of sample contamination'
    power = '5mW pump 5mW probe' # can also set any of these to None
    pressure = 1.51e-8

    t0 = 8.26

    log_folder = f'C:\\Users\\247 HP Pavilion\\Documents\\{foldername}\\{filename}\\'
    data_folder = f'/data/{foldername}/{filename}'
    continue_run = True

    parameters = [[19.6472, 25.8, 10],[19.6475, 19.9, 10],[19.6478, 19.9, 10],[19.6481, 19.9, 10]]
    
    delays_col = 1

    # spectrometer
    spec_integration_time = 5000 # 20000 # 10000 # 20000 
    spec_averages = 5

    # delay stage
    delay_actuator_ID = 27000853

    # rotation stage
    compressor_COM_port = 'COM3'
    wedges_COM_port = 'COM6'
    
    rotation_COM_port = compressor_COM_port
    # rotation_axis = 3# if for the wedge controller's 3rd axis, controlling the 200 stage for now
    rotation_axis = 2# if for the compressor com port to control the rotational stage

    specs = oo.spectrometers()
    if len(specs.devices) > 1:
        print(f'Spectrometer is set to be {specs.devices[0]}, is that correct? sucks if not lol')

    # # arduion
    # arduino_COM_port = 'COM3'

    compressor_positions = check_start_positions(compressor_COM_port)
    # print(compressor_positions)
    wedge_positions = check_start_positions(wedges_COM_port)
    # print(wedge_positions)

    # run scan
    if continue_run or not os.path.exists(log_folder):
        wait_times = [parameter_set[2] for parameter_set in parameters]
        if min(wait_times) > 2*spec_integration_time*spec_averages/1000000:
            logger.post_log(filename, data_folder, log_folder, note=note, pressure=pressure, power=power, t0=t0,delays_col=delays_col,
                             parameters=parameters, wedge_positions = wedge_positions, compressor_positions = compressor_positions)
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            parameter_scanny(data_folder, log_folder, delay_actuator_ID, rotation_COM_port, rotation_axis, parameters, spec_integration_time, spec_averages)
        else:
            print('Failed to start scan: spectrum collection too long relative to wait time')
    else:
        print('Failed to start scan: data folder already exists and continue_run is set to False')


main()