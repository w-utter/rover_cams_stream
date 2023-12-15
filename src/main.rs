static mut UVC_CONTEXT: Option<uvc::Context<'static>> = None;

/*
struct CamerasStream;

impl CamerasStream {
    pub fn start() {
        refresh_context(uvc::Context::new().unwrap());
        //TODO: add an api and move out of a main fn
    }
}

impl Drop for CamerasStream {
    fn drop(&mut self) {
        unsafe {UVC_CONTEXT = None};
    }
}
*/

macro_rules! map_recv {
    ($recv:expr) => {
        match $recv {
            Ok(ok) => Some(ok),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            _ => break,
        }
    };
}

macro_rules! map_recv_ip {
    ($recv:expr) => {
        match $recv {
            Ok(ok) => Some(ok),
            _ => None,
        }
    };
}

#[inline]
fn get_context() -> &'static uvc::Context<'static> {
    unsafe { UVC_CONTEXT.as_ref().unwrap() }
}

#[inline]
fn refresh_context(context: uvc::Context<'static>) -> &'static uvc::Context<'static> {
    unsafe {
        UVC_CONTEXT = Some(context);
    }
    get_context()
}

use rusb::UsbContext;

#[tokio::main]
async fn main() {
    refresh_context(uvc::Context::new().unwrap());

    let (logging_send, logging_recv) = std::sync::mpsc::channel();
    let (framerates_send, framerates_recv) = std::sync::mpsc::channel();
    let (io_ev_send, io_ev_recv) = std::sync::mpsc::channel();
    let (min_io_send, min_io_recv) = std::sync::mpsc::channel();

    let io_thread = IoThread::init(logging_send.clone(), io_ev_send, min_io_send);
    let logging_thread = LoggingThread::init(logging_recv, framerates_recv);

    let (encoding_send, encoding_recv) = std::sync::mpsc::channel();
    let (comms_send, comms_recv) = std::sync::mpsc::channel();

    let manager_thread = ManagerThread::init(
        io_ev_recv,
        logging_send.clone(),
        comms_recv,
        framerates_send,
        encoding_send,
    );
    //let comms_thread = CommunicationThread::init(encoding_recv, logging_send, min_io_recv, comms_send).await;

    std::thread::spawn(move || {
        manager_thread.spin();
    });

    std::thread::spawn(move || {
        logging_thread.spin();
    });

    //surely this gets put onto the async q
    //comms_thread.spin().await;

    io_thread.spin()
}

/*
struct CommConnection {
    connection: std::sync::Arc<webrtc::peer_connection::RTCPeerConnection>,
    cmds_cb: std::sync::Arc<webrtc::data_channel::RTCDataChannel>,
    logging_cb: std::sync::Arc<webrtc::data_channel::RTCDataChannel>,
}

impl CommConnection {
    async fn new(cmds: std::sync::mpsc::Sender<CommunicationCommand>, logging: std::sync::mpsc::Sender<String>) -> Self {
        let mut media = webrtc::api::media_engine::MediaEngine::default();
        let mut reg = webrtc::interceptor::registry::Registry::new();
        reg = webrtc::api::interceptor_registry::register_default_interceptors(reg, &mut media).unwrap();
        let api = webrtc::api::APIBuilder::new().with_media_engine(media).with_interceptor_registry(reg).build();
        let config = webrtc::peer_connection::configuration::RTCConfiguration::default();
        let connection = std::sync::Arc::new(api.new_peer_connection(config).await.unwrap());
        let logging = std::sync::Arc::new(logging);

        connection.on_peer_connection_state_change(Box::new(|state| {
            use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
            match state {
                //TODO
                RTCPeerConnectionState::Failed => {
                }
                RTCPeerConnectionState::Connecting => {
                }
                RTCPeerConnectionState::New => {
                }
                _ => {}
            }
            Box::pin(async {})
        }));
        let cmds = std::sync::Arc::new(cmds);

        let cmds_channel = connection.create_data_channel("cmds", None).await.unwrap();

        let log = logging.clone();
        cmds_channel.on_message(Box::new(move |msg| {
                if let Some(msg) = CommunicationCommand::from_data_channel_message(&msg) {
                    cmds.send(msg).unwrap();
                }else {
                    log.send(format!("{:?}", msg)).unwrap();
                    //TODO?: NACK the lost msg
                }

                Box::pin(async {})
        }));

        let logging_channel = connection.create_data_channel("log", None).await.unwrap();

        //TODO: add more functionality to the logging_channel

        let conn = connection.clone();

        connection.on_ice_candidate(Box::new(move |maybe_candidate| {
            let conn = conn.clone();
            Box::pin(async move {
                if let Some(Ok(candidate)) = maybe_candidate.map(|candidate| candidate.to_json()) {
                    conn.add_ice_candidate(candidate).await.unwrap()
                }
            })
        }));

        //let offer = connection.create_offer(None).await.unwrap();
        let offer = get_rtc_session_description();
        connection.set_remote_description(offer).await.unwrap();
        let answer = connection.create_answer(None).await.unwrap();
        let mut gather = connection.gathering_complete_promise().await;
        connection.set_local_description(answer).await.unwrap();
        gather.recv().await;


        Self {
            connection,
            cmds_cb: cmds_channel,
            logging_cb: logging_channel,
        }
    }

    async fn return_comm_cmd(&self, msg: ReturnCommunicationCommand) {
        if let Err(_) = self.cmds_cb.send(msg.serialize()).await {
            //TODO?: log
        }
    }


}
*/

#[derive(Clone)]
enum ReturnCommunicationCommand {
    CameraJoined(DeviceIndex),
    CameraLeft(DeviceIndex),
    Ack(CommunicationCommand),
    //Nack(webrtc::data_channel::data_channel_message::DataChannelMessage),
}

/*
impl ReturnCommunicationCommand {
    fn serialize(&self) -> &bytes::Bytes {
        //TODO:
        todo!()
    }
}
*/

fn read_stdin() -> Result<String, std::io::Error> {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(line.trim().to_owned())
}

/*
fn get_rtc_session_description() -> webrtc::peer_connection::sdp::session_description::RTCSessionDescription {
    use base64::Engine;
    let bytes = base64::prelude::BASE64_STANDARD.decode(read_stdin().unwrap()).unwrap();
    webrtc::peer_connection::sdp::session_description::RTCSessionDescription::offer(String::from_utf8(bytes).unwrap()).unwrap()
}
*/

const BASE_IP: &'static str = "";

const IP: &'static str = "127.0.0.1";

/*
struct StreamConnection {
    track: std::sync::Arc<webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample>,
    sender: std::sync::Arc<webrtc::rtp_transceiver::rtp_sender::RTCRtpSender>
}

impl StreamConnection {
    async fn new(connection: &std::sync::Arc<webrtc::peer_connection::RTCPeerConnection>, _idx: &DeviceIndex) -> Self {
        let track = std::sync::Arc::new(
            webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample::new(
                webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability {
                    mime_type: webrtc::api::media_engine::MIME_TYPE_AV1.to_owned(),
                    ..Default::default()
                },
                "video".to_string(),
                "webrtc-rs".to_string(),
            )
        );

        let sender = connection.add_track(track.clone()).await.unwrap();

        Self {
            track,
            sender,
        }
    }
}
*/

/*
struct CommunicationThread {
    frames: std::sync::mpsc::Receiver<EncodedFrame>,
    //streams: std::collections::HashMap<DeviceIndex, StreamConnection>,
    comms: CommConnection,
    logging: std::sync::mpsc::Sender<String>,
    io_events: std::sync::mpsc::Receiver<MinIoEvent>,
}
*/

struct MinIoEvent {
    index: DeviceIndex,
    kind: MinIoEventKind,
}

enum MinIoEventKind {
    Joined,
    Left,
}

impl MinIoEventKind {
    fn from_device_event(ev: &DeviceIoEventKind) -> Option<Self> {
        Some(match ev {
            DeviceIoEventKind::Joined(_) => Self::Joined,
            DeviceIoEventKind::Left => Self::Left,
        })
    }
}

/*
impl CommunicationThread {
    async fn init(frames: std::sync::mpsc::Receiver<EncodedFrame>, logging: std::sync::mpsc::Sender<String>, io_events: std::sync::mpsc::Receiver<MinIoEvent>, comms: std::sync::mpsc::Sender<CommunicationCommand>) -> Self {
        let comms = CommConnection::new(comms, logging.clone()).await;
        let streams = std::collections::HashMap::with_capacity(MAX_CAMS);

        Self {
            frames,
            streams,
            comms,
            logging,
            io_events,
        }
    }

    fn log(&self, msg: String) {
        self.logging.send(msg).unwrap();
    }

    async fn add_stream(&mut self, index: DeviceIndex) {
        if let Some(_) = self.streams.get(&index) {
            self.log(format!("stream already exists with idx {:?}", index));
        }else {
            let conn = StreamConnection::new(&self.comms.connection, &index).await;
            self.streams.insert(index, conn);
        }
        self.reinstate_connection().await;
    }

    async fn remove_stream(&mut self, index: DeviceIndex) {
        if let Some(stream) = self.streams.remove(&index) {
            self.comms.connection.remove_track(stream.track.clone()).await.unwrap();
        }else {
            self.log(format!("stream does not exist with idx {:?} despite being disconnected", index));
        }
        self.reinstate_connection().await;
    }

    //FIXME: i dont know whether either implementation of setting up the offer is correct as there
    //arent any available examples for the server directing events.
    async fn reinstate_connection(&self) {
        let connection = &self.comms.connection;
        let offer = connection.create_offer(None).await.unwrap();
        connection.set_remote_description(offer).await.unwrap();
        let mut gathering = connection.gathering_complete_promise().await;
        gathering.recv().await;
    }

    async fn send_packet(&self, pkt: EncodedFrame) {
        let EncodedFrame {
            raw,
            frame_type,
            enc_info,
            quantization_param,
            idx,
        } = pkt;

        //FIXME: provide more info from EncodedFrame

        let stream = &self.streams[&idx];

        stream.track.write_sample(&webrtc::media::Sample {
            data: bytes::Bytes::copy_from_slice(&*raw),
            ..Default::default()
        }).await.unwrap();
    }

    async fn spin(mut self) -> u8 {
        loop {
            if let Some(pkt) = map_recv!(self.frames.try_recv()) {
                self.send_packet(pkt).await;
            }
            if let Some(io_ev) = map_recv!(self.io_events.try_recv()) {
                match io_ev.kind {
                    MinIoEventKind::Joined => self.add_stream(io_ev.index).await,
                    MinIoEventKind::Left => self.remove_stream(io_ev.index).await,
                }
            }
            //if let Some(comm_ev) = self.
        }
        0
    }
}
*/

struct LoggingThread {
    log: std::sync::mpsc::Receiver<String>,
    framerates: std::sync::mpsc::Receiver<Box<[PipelineFramerates]>>,
}

impl LoggingThread {
    fn init(
        recv: std::sync::mpsc::Receiver<String>,
        framerates: std::sync::mpsc::Receiver<Box<[PipelineFramerates]>>,
    ) -> Self {
        Self {
            log: recv,
            framerates,
        }
    }

    fn spin(self) {
        loop {
            if let Some(msg) = map_recv!(self.log.try_recv()) {
                println!("{msg}");
            }
            /*
            if let Some(framerates) = map_recv!(self.framerates.try_recv()) {
                if !framerates.is_empty() {
                    println!("{:?}", framerates);
                }
            }
            */
        }
    }
}

struct ManagerThread {
    io_events: std::sync::mpsc::Receiver<DeviceIoEvent>,
    comms_cmds: std::sync::mpsc::Receiver<CommunicationCommand>,
    logging: std::sync::mpsc::Sender<String>,
    current_pipelines: std::collections::HashMap<DeviceIndex, PipelineManager<()>>,
    framerates: std::sync::mpsc::Sender<Box<[PipelineFramerates]>>,
    encoding: std::sync::mpsc::Sender<EncodedFrame>,
}

#[derive(Clone)]
enum CommunicationCommand {
    Unknown,
}

/*
impl CommunicationCommand {
    fn from_data_channel_message(
        msg: &webrtc::data_channel::data_channel_message::DataChannelMessage,
    ) -> Option<Self> {
        None
        //CommunicationCommand::Unknown
    }
}
*/

struct FrameDimensions {
    width: u32,
    height: u32,
}

struct CameraThread<'a> {
    cam_cmds: std::sync::mpsc::Receiver<CameraCommand>,
    logging: std::sync::mpsc::Sender<String>,
    raw_frame_cb: std::sync::mpsc::Sender<()>,
    send_frame: std::sync::mpsc::Sender<RawFrame>,
    send_dimensions: std::sync::mpsc::Sender<FrameDimensions>,
    dev: uvc::Device<'a>,
    device_index: DeviceIndex,
}

#[non_exhaustive]
enum CameraCommand {
    Close,
}

struct RawFrame {
    data: Box<[u8]>,
}

impl RawFrame {
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl RawFrame {
    fn from_uvc(frame: &uvc::Frame) -> Result<RawFrame, Box<uvc::Error>> {
        Ok(RawFrame {
            data: frame.to_rgb()?.to_bytes().to_owned().into_boxed_slice(),
        })
    }
}

impl<'a> CameraThread<'a> {
    fn init(
        cam_cmds: std::sync::mpsc::Receiver<CameraCommand>,
        frame_notif: std::sync::mpsc::Sender<()>,
        send_frame: std::sync::mpsc::Sender<RawFrame>,
        send_dimensions: std::sync::mpsc::Sender<FrameDimensions>,
        logging: std::sync::mpsc::Sender<String>,
        dev: uvc::Device<'a>,
        dev_idx: DeviceIndex,
    ) -> Self {
        Self {
            logging,
            raw_frame_cb: frame_notif,
            cam_cmds,
            send_frame,
            dev,
            device_index: dev_idx,
            send_dimensions,
        }
    }

    fn spin(mut self) {
        let devh = self.dev.open().expect("cannot open device");
        let preferred_format = devh
            .get_preferred_format(|x, y| {
                if x.fps >= y.fps && x.width * x.height >= y.width * y.height {
                    x
                } else {
                    y
                }
            })
            .unwrap();

        let mut streamh = None;

        while streamh.is_none() {
            if let Ok(handle) = devh.get_stream_handle_with_format(preferred_format) {
                streamh = Some(handle);
            }
        }
        let mut streamh = streamh.unwrap();

        self.send_dimensions.send(FrameDimensions {
            width: preferred_format.width,
            height: preferred_format.height,
        });

        let logging = self.logging;
        let send_frame = self.send_frame;
        let frame_cb = self.raw_frame_cb;

        let stream = streamh
            .start_stream(move |frame| match RawFrame::from_uvc(frame) {
                Ok(raw) => {
                    send_frame.send(raw).unwrap();
                    frame_cb.send(()).unwrap();
                }
                Err(e) => logging
                    .send(format!("could not conv frame: {:?}", e))
                    .unwrap(),
            })
            .unwrap();

        loop {
            if let Some(cmd) = map_recv!(self.cam_cmds.try_recv()) {
                match cmd {
                    //TODO
                    CameraCommand::Close => break,
                    _ => {}
                }
            }
        }
    }

    fn send_raw_frame(&self, frame: RawFrame) {
        self.send_frame.send(frame).unwrap();
    }

    fn log(&self, msg: String) {
        self.logging.send(msg).unwrap()
    }
}

struct EncoderThread {
    cmds: std::sync::mpsc::Receiver<EncodingCommand>,
    logging: std::sync::mpsc::Sender<String>,
    frame_cb: std::sync::mpsc::Sender<()>,
    recv_frame_dimensions: std::sync::mpsc::Receiver<FrameDimensions>,
    recv_raw_frames: std::sync::mpsc::Receiver<RawFrame>,
    send_enc_frames: std::sync::mpsc::Sender<EncodedFrame>,
    dev_idx: DeviceIndex,
    cfg: rav1e::Config,
    ctx: rav1e::Context<u8>,
    enc: rav1e::EncoderConfig,

    //hw enc
    //surface: cros_libva::Surface<()>,
    config: cros_libva::Config,
    display: std::rc::Rc<cros_libva::Display>,
    seq_fields: cros_libva::H264EncSeqFields,
}

// NOTE: there can actually be multiple frames bundled with this one
// (dec should handle it however)
struct EncodedFrame {
    raw: Box<[u8]>,
    frame_type: rav1e::prelude::FrameType,
    enc_info: rav1e::data::EncoderStats,
    quantization_param: u8,
    idx: DeviceIndex,
}

impl EncodedFrame {
    fn from_rav1e(pkt: rav1e::prelude::Packet<u8>, idx: &DeviceIndex) -> Self {
        let rav1e::prelude::Packet {
            data,
            frame_type,
            qp,
            enc_stats,
            ..
        } = pkt;

        Self {
            raw: data.into_boxed_slice(),
            frame_type,
            enc_info: enc_stats,
            quantization_param: qp,
            idx: idx.clone(),
        }
    }
}

struct HWEncParams<T: cros_libva::SurfaceMemoryDescriptor> {
    surface: cros_libva::Surface<T>,
    context: std::rc::Rc<cros_libva::Context>,
    width: u32,
    height: u32,
}

impl EncoderThread {
    fn init(
        cmds: std::sync::mpsc::Receiver<EncodingCommand>,
        frame_notif: std::sync::mpsc::Sender<()>,
        recv_raw_frames: std::sync::mpsc::Receiver<RawFrame>,
        recv_dimensions: std::sync::mpsc::Receiver<FrameDimensions>,
        logging: std::sync::mpsc::Sender<String>,
        send_enc_frames: std::sync::mpsc::Sender<EncodedFrame>,
        dev_idx: &DeviceIndex,
    ) -> Self {
        //let speed_settings =
        let enc = rav1e::EncoderConfig {
            width: 64,
            height: 96,
            speed_settings: rav1e::config::SpeedSettings::from_preset(10),
            low_latency: true,
            ..Default::default()
        };
        let cfg = rav1e::Config::default()
            .with_encoder_config(enc.clone())
            .with_threads(4);
        let ctx = cfg.new_context::<u8>().unwrap();

        let mut attrs = vec![cros_libva::VAConfigAttrib {
            type_: cros_libva::VAConfigAttribType::VAConfigAttribRTFormat,
            value: 0,
        }];

        let display = cros_libva::Display::open().unwrap();
        let profile = cros_libva::VAProfile::VAProfileH264ConstrainedBaseline;
        let entrypoint = cros_libva::VAEntrypoint::VAEntrypointEncSliceLP;
        let format = cros_libva::constants::VA_RT_FORMAT_YUV420;

        display
            .get_config_attributes(profile, entrypoint, &mut attrs)
            .unwrap();

        let config = display.create_config(attrs, profile, entrypoint).unwrap();
        let seq_fields = cros_libva::H264EncSeqFields::new(1, 1, 0, 0, 0, 1, 0, 2, 0);

        //out all of the available codecs for hardware encoding
        let available_formats = display.query_image_formats().unwrap();
        println!("{:?}", available_formats);

        let fmt = available_formats
            .into_iter()
            .find(|f| f.fourcc == cros_libva::constants::VA_FOURCC_NV12)
            .unwrap();

        Self {
            cmds,
            frame_cb: frame_notif,
            recv_raw_frames,
            recv_frame_dimensions: recv_dimensions,
            config,
            display,
            seq_fields,
            logging,
            send_enc_frames,
            dev_idx: dev_idx.clone(),
            enc,
            cfg,
            ctx,
        }
    }

    fn setup_encoder<T: cros_libva::SurfaceMemoryDescriptor>(
        &self,
        width: u32,
        height: u32,
    ) -> HWEncParams<T> {
        let mut surfaces = self
            .display
            .create_surfaces(
                format,
                None,
                width,
                height,
                Some(cros_libva::UsageHint::USAGE_HINT_ENCODER),
                vec![],
            )
            .unwrap();

        let context = self
            .display
            .create_context(&self.config, width, height, Some(&surfaces), true)
            .unwrap();
        let surface = surfaces.pop().unwrap();
    }

    fn setup_picture<T: cros_libva::SurfaceMemoryDescriptor, U: cros_libva::PictureState, V>(
        &self,
        params: &HWEncParams<T>,
        buf: RawFrame,
    ) -> cros_libva::Picture<U, V> {
        let sps =
            cros_libva::BufferType::EncSequenceParameter(cros_libva::EncSequenceParameter::H264(
                cros_libva::EncSequenceParameterBufferH264::new(
                    0,
                    10,
                    10,
                    30,
                    1,
                    0,
                    1,
                    params.width / 16 as u16,
                    params.height / 16 as u16,
                    &self.seq_fields,
                    0,
                    0,
                    0,
                    0,
                    0,
                    [0; 256],
                    None,
                    Some(cros_libva::H264VuiFields::new(1, 1, 0, 0, 0, 1, 0, 0)),
                    255,
                    1,
                    1,
                    1,
                    60,
                ),
            ));
        let ref_frames: [cros_libva::PictureH264; 16] = (0..16)
            .map(|_| {
                cros_libva::PictureH264::new(
                    cros_libva::constants::VA_INVALID_ID,
                    0,
                    cros_libva::constants::VA_INVALID_SURFACE,
                    0,
                    0,
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let coded_buffer = params.context.create_enc_coded(buf.len()).unwrap();

        let pps = cros_libva::BufferType::EncPictureParameter(
            cros_libva::EncPictureParameter::H264(cros_libva::EncPictureParameterBufferH264::new(
                cros_libva::PictureH264::new(params.surface.id(), 0, 0, 0, 0),
                ref_frames,
                coded_buffer.id(),
                0,
                0,
                0,
                0,
                26,
                0,
                0,
                0,
                0,
                &cros_libva::H264EncPicFields::new(1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
            )),
        );

        let ref_pic_list_0: [cros_libva::PictureH264; 32] = (0..32)
            .map(|_| {
                cros_libva::PictureH264::new(
                    cros_libva::constants::VA_INVALID_ID,
                    0,
                    cros_libva::constants::VA_INVALID_SURFACE,
                    0,
                    0,
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let ref_pic_list_1: [cros_libva::PictureH264; 32] = (0..32)
            .map(|_| {
                cros_libva::PictureH264::new(
                    cros_libva::constants::VA_INVALID_ID,
                    0,
                    cros_libva::constants::VA_INVALID_SURFACE,
                    0,
                    0,
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let slice = cros_libva::BufferType::EncSliceParameter(cros_libva::EncSliceParameter::H264(
            cros_libva::EncSliceParameterBufferH264::new(
                0,
                ((params.width / 16) * (params.height / 16)) as u32,
                cros_libva::constants::VA_INVALID_ID,
                2, // I
                0,
                1,
                0,
                0,
                [0, 0],
                1,
                0,
                0,
                0,
                ref_pic_list_0,
                ref_pic_list_1,
                0,
                0,
                0,
                [0; 32],
                [0; 32],
                0,
                [[0; 2]; 32],
                [[0; 2]; 32],
                0,
                [0; 32],
                [0; 32],
                0,
                [[0; 2]; 32],
                [[0; 2]; 32],
                0,
                0,
                0,
                2,
                2,
            ),
        ));
        let data = cros_libva::BufferType::SliceData(buf.data);
        let pps = params.context.create_buffer(pps).unwrap();
        let sps = params.context.create_buffer(sps).unwrap();
        let slice = params.context.create_buffer(slice).unwrap();
        let data = params.context.create_buffer(data).unwrap();
        let mut picture = cros_libva::Picture::new(0, params.context.clone(), params.surface);
        picture.add_buffer(pps);
        picture.add_buffer(sps);
        picture.add_buffer(slice);
        picture.add_buffer(data);
        picture
    }

    // FIXME?: it might be beneficial to have an extra field to RawFrame which allows for switching
    // the encoder config (e.g if the dimensions of the image change) to be able to switch the
    // encoding settings

    fn spin(mut self) {
        const DEFAULT_WIDTH: u32 = 640;
        const DEFAULT_HEIGHT: u32 = 480;

        let mut encoder_params = None;
        let mut picture = None;
        let socket = std::net::UdpSocket::bind("127.0.0.1::2048").unwrap();
        socket.connect("127.0.0.1:4031");
        //TODO: switch to this
        //socket.connect(&format!("127.0.0.1:{}", self.dev_idx.0));
        loop {
            if let Some(cmd) = map_recv!(self.cmds.try_recv()) {
                match cmd {
                    EncodingCommand::Close => break,
                    _ => {}
                }
            }

            if let Some(FrameDimensions { width, height }) =
                map_recv!(self.recv_frame_dimensions.try_recv())
            {
                encoder_params = Some(self.setup_encoder(width, height));
            }

            if let Some(params) = encoder_params.as_ref() {
                if let Some(raw_frame) = map_recv!(self.recv_raw_frames.try_recv()) {
                    let pic = self.setup_picture(params, raw_frame);
                    let picture = pic
                        .begin()
                        .unwrap()
                        .render()
                        .unwrap()
                        .end()
                        .unwrap()
                        .sync()
                        .map_err(|_, e| e)
                        .unwrap();
                    let img = picture.derive_image((params.width, params.height)).unwrap();
                    let packet = rtp_rs::RtpPacketBuilder::new().payload(&*img).build();
                    socket.send(&packet);
                }
            }

            /* old rav1e encoding
            if let Some(pkt) = self.encoded_packet() {
                let packet_number = pkt.input_frameno;
                self.send_encoded_frame(EncodedFrame::from_rav1e(pkt, &self.dev_idx));
                self.frame_cb.send(()).unwrap();
                if packet_number > 6000 {
                    // removing the buf of frames every couple min or so
                    // to not have an issue with the buf being filled up
                    self.flush();
                }
            }

            if let Some(raw) = map_recv!(self.recv_raw_frames.try_recv()) {
                let mut frame = self.ctx.new_frame();

                for plane in &mut frame.planes {
                    let stride = (self.enc.width + plane.cfg.xdec) >> plane.cfg.xdec;
                    plane.copy_from_raw_u8(&raw.data, stride, 1);
                }

                self.encode_frame(frame)
            }
            */
        }
    }

    fn encode_frame(&mut self, frame: rav1e::Frame<u8>) {
        if let Err(err) = self.ctx.send_frame(frame) {
            match err {
                //FIXME: log here
                _ => {}
            }
        }
    }

    fn encoded_packet(&mut self) -> Option<rav1e::Packet<u8>> {
        use rav1e::EncoderStatus as ES;
        match self.ctx.receive_packet() {
            Ok(pkt) => Some(pkt),
            Err(e) => {
                match e {
                    ES::LimitReached => {
                        self.log("encoder buf somehow got filled; flushing buffer".to_string());
                        self.flush()
                    }
                    ES::NeedMoreData => {
                        //self.log("not enough data was sent to the encoder".to_string());
                    }
                    _ => {}
                }

                None
            }
        }
    }

    fn flush(&mut self) {
        self.ctx.flush()
    }

    fn send_encoded_frame(&self, frame: EncodedFrame) {
        self.send_enc_frames.send(frame).unwrap();
    }

    fn log(&self, msg: String) {
        self.logging.send(msg).unwrap();
    }
}

#[non_exhaustive]
enum EncodingCommand {
    Close,
}

struct PipelineManager<T> {
    cam_cmds: std::sync::mpsc::Sender<CameraCommand>,
    enc_cmds: std::sync::mpsc::Sender<EncodingCommand>,
    cam_handle: Option<std::thread::JoinHandle<T>>,
    enc_handle: Option<std::thread::JoinHandle<T>>,
    raw_frame_notif: std::sync::mpsc::Receiver<()>,
    enc_frame_notif: std::sync::mpsc::Receiver<()>,
    time_since_last_raw: Time,
    time_since_last_enc: Time,
    last_raw_framerate: u8,
    last_enc_framerate: u8,
}

#[derive(Debug)]
struct FpsRates {
    raw: u8,
    enc: u8,
}

impl<T> PipelineManager<T> {
    fn fps(&mut self, now: &Time) -> FpsRates {
        //NOTE: most of this stuff should be pushed into process_framerate
        let raw = if let Some(raw_framerate) =
            process_framerate(&self.raw_frame_notif, &self.time_since_last_raw, &now)
        {
            self.time_since_last_raw = now.clone();
            self.last_raw_framerate = raw_framerate;
            raw_framerate
        } else {
            self.last_raw_framerate
        };

        let enc = if let Some(enc_framerate) =
            process_framerate(&self.enc_frame_notif, &self.time_since_last_enc, &now)
        {
            self.time_since_last_enc = now.clone();
            self.last_enc_framerate = enc_framerate;
            enc_framerate
        } else {
            self.last_enc_framerate
        };

        FpsRates { raw, enc }
    }

    #[inline]
    fn close_cam(&mut self) {
        self.cam_cmds.send(CameraCommand::Close).unwrap();
        if let Some(handle) = self.cam_handle.take() {
            handle.join().unwrap();
        }
    }

    // FIXME: trying to run this on macos will not work.
    // see https://github.com/libuvc/libuvc/pull/251
    // need to change the vendor for the bindings + refactor some

    #[inline]
    fn close_enc(&mut self) {
        self.enc_cmds.send(EncodingCommand::Close).unwrap();
        if let Some(handle) = self.enc_handle.take() {
            handle.join().unwrap();
        }
    }

    #[inline]
    fn close(&mut self) {
        self.close_cam();
        self.close_enc();
    }

    fn instantiate(
        cam_cmds: std::sync::mpsc::Sender<CameraCommand>,
        enc_cmds: std::sync::mpsc::Sender<EncodingCommand>,
        cam_handle: Option<std::thread::JoinHandle<T>>,
        enc_handle: Option<std::thread::JoinHandle<T>>,
        raw_frame_notif: std::sync::mpsc::Receiver<()>,
        enc_frame_notif: std::sync::mpsc::Receiver<()>,
    ) -> Self {
        let now = Time::now();
        Self {
            last_enc_framerate: 0,
            last_raw_framerate: 0,
            time_since_last_enc: now.clone(),
            time_since_last_raw: now,
            cam_cmds,
            enc_cmds,
            cam_handle,
            enc_handle,
            raw_frame_notif,
            enc_frame_notif,
        }
    }
}

fn process_framerate(
    recv: &std::sync::mpsc::Receiver<()>,
    last_time: &Time,
    now: &Time,
) -> Option<u8> {
    if let Ok(_) = recv.try_recv() {
        let diff = (now.clone().as_millis() - last_time.clone().as_millis()) as u16;
        if diff > 1000 {
            return Some(1);
        } else if diff < 4 {
            return Some(255);
        }
        Some((1000 / diff) as u8)
    } else {
        None
    }
}

const MAX_CAMS: usize = 7;

impl ManagerThread {
    fn init(
        io_events: std::sync::mpsc::Receiver<DeviceIoEvent>,
        logging: std::sync::mpsc::Sender<String>,
        comms_cmds: std::sync::mpsc::Receiver<CommunicationCommand>,
        framerates: std::sync::mpsc::Sender<Box<[PipelineFramerates]>>,
        encoding: std::sync::mpsc::Sender<EncodedFrame>,
    ) -> Self {
        Self {
            io_events,
            logging,
            comms_cmds,
            current_pipelines: std::collections::HashMap::with_capacity(MAX_CAMS),
            framerates,
            encoding,
        }
    }

    const TRACK_FRAMERATES: bool = true;

    fn spin(mut self) {
        loop {
            if let Some(io_event) = map_recv!(self.io_events.try_recv()).take() {
                if let DeviceIoEventKind::Joined(dev) = io_event.kind {
                    self.request_open_pipeline(io_event.dev, dev)
                } else {
                    self.request_close_pipeline(io_event.dev);
                }
            }

            if let Some(comms_cmd) = map_recv!(self.comms_cmds.try_recv()) {
                //TODO
                match comms_cmd {
                    _ => {}
                }
            }

            if Self::TRACK_FRAMERATES {
                let fps_rates = self.get_framerates();
                self.framerates.send(fps_rates).unwrap();
            }
        }
    }

    #[inline]
    fn get_framerates(&mut self) -> Box<[PipelineFramerates]> {
        let now = Time::now();
        self.current_pipelines
            .iter_mut()
            .map(|(dev, pipeline)| {
                PipelineFramerates::from_idx_and_fps(dev.clone(), pipeline.fps(&now))
            })
            .collect::<Box<_>>()
    }

    #[inline]
    fn new_camera_thread(
        &self,
        cam_cmds: std::sync::mpsc::Receiver<CameraCommand>,
        send_dimensions: std::sync::mpsc::Receiver<FrameDimensions>,
        frame_notif: std::sync::mpsc::Sender<()>,
        recv_frame: std::sync::mpsc::Sender<RawFrame>,
        dev: &MinDevice,
        uvc_dev: uvc::Device<'static>,
    ) -> Option<std::thread::JoinHandle<()>> {
        let cam_thread = CameraThread::init(
            cam_cmds,
            frame_notif,
            recv_frame,
            send_dimensions,
            self.logging.clone(),
            uvc_dev,
            dev.index.clone(),
        );
        Some(std::thread::spawn(move || {
            cam_thread.spin();
        }))
    }

    #[inline]
    fn new_transcoding_thread(
        &self,
        enc_cmds: std::sync::mpsc::Receiver<EncodingCommand>,
        frame_notif: std::sync::mpsc::Sender<()>,
        send_frame: std::sync::mpsc::Receiver<RawFrame>,
        recv_dimensions: std::sync::mpsc::Receiver<FrameDimensions>,
        dev: &MinDevice,
    ) -> Option<std::thread::JoinHandle<()>> {
        let enc_thread = EncoderThread::init(
            enc_cmds,
            frame_notif,
            send_frame,
            recv_dimensions,
            self.logging.clone(),
            self.encoding.clone(),
            &dev.index,
        );

        Some(std::thread::spawn(move || {
            enc_thread.spin();
        }))
    }

    fn request_open_pipeline(&mut self, dev: MinDevice, uvc_dev: uvc::Device<'static>) {
        let idx = dev.idx();

        let (close_cam_send, close_cam_recv) = std::sync::mpsc::channel();
        let (close_enc_send, close_enc_recv) = std::sync::mpsc::channel();

        let (cam_notif_send, cam_notif_recv) = std::sync::mpsc::channel();
        let (enc_notif_send, enc_notif_recv) = std::sync::mpsc::channel();

        let (frame_send, frame_recv) = std::sync::mpsc::channel();
        let (send_dimensions, recv_dimensions) = std::sync::mpsc::channel();

        let cam_handle = self.new_camera_thread(
            close_cam_recv,
            send_dimensions,
            cam_notif_send,
            frame_send,
            &dev,
            uvc_dev,
        );
        let enc_handle = self.new_transcoding_thread(
            close_enc_recv,
            enc_notif_send,
            frame_recv,
            recv_dimensions,
            &dev,
        );

        if let Some(pipeline) = self.current_pipelines.get_mut(idx) {
            *pipeline = PipelineManager::instantiate(
                close_cam_send,
                close_enc_send,
                cam_handle,
                enc_handle,
                cam_notif_recv,
                enc_notif_recv,
            );
            self.log(format!(
                "pipeline already exists with idx {:?}, replacing the current manager",
                idx
            ));
        } else {
            self.current_pipelines.insert(
                dev.index,
                PipelineManager::instantiate(
                    close_cam_send,
                    close_enc_send,
                    cam_handle,
                    enc_handle,
                    cam_notif_recv,
                    enc_notif_recv,
                ),
            );
        }
    }

    fn request_close_pipeline(&mut self, dev: MinDevice) -> () {
        if let Some(mut pipeline) = self.current_pipelines.remove(dev.idx()) {
            pipeline.close();
        } else {
            self.log(format!(
                "pipeline with idx {:?} does not exist within the map despite being disconnected",
                dev.idx()
            ))
        }
    }

    fn log(&self, msg: String) {
        self.logging.send(msg).unwrap();
    }
}

#[derive(Debug)]
struct PipelineFramerates {
    fps: FpsRates,
    idx: DeviceIndex,
}

impl PipelineFramerates {
    const fn from_idx_and_fps(idx: DeviceIndex, fps: FpsRates) -> Self {
        Self { idx, fps }
    }
}

#[derive(Debug)]
struct MinDevice {
    vendor_id: Option<i32>,
    product_id: Option<i32>,
    index: DeviceIndex,
}

#[repr(transparent)]
#[derive(Debug, Hash, Clone, PartialEq, Eq)]
struct DeviceIndex(u16);

#[inline]
fn is_special_cam(dev: &rusb::DeviceDescriptor, int: &rusb::InterfaceDescriptor) -> bool {
    dev.vendor_id() == 0x199e
        && (dev.product_id() == 0x8102 || dev.product_id() == 0x8102)
        && (int.class_code() == 255 && int.sub_class_code() == 2)
}

#[inline]
fn is_cam(int: &rusb::InterfaceDescriptor) -> bool {
    int.class_code() == 14 && int.sub_class_code() == 2
}

impl MinDevice {
    fn from_libusb_device<T>(device: rusb::Device<T>) -> Option<Self>
    where
        T: rusb::UsbContext,
    {
        let config = device.config_descriptor(0).ok()?;
        let descriptor = device.device_descriptor().ok()?;

        //TIS cams are not uvc
        if is_tis_cam(&descriptor) {
            return None;
        }

        for interface in config.interfaces() {
            for alt_setting in interface.descriptors() {
                if is_special_cam(&descriptor, &alt_setting) || is_cam(&alt_setting) {
                    return Some(Self {
                        index: DeviceIndex(
                            (device.bus_number() as u16) | ((device.address() as u16) << 8),
                        ),
                        vendor_id: Some(descriptor.vendor_id() as i32),
                        product_id: Some(descriptor.product_id() as i32),
                    });
                }
            }
        }
        None
    }

    fn idx(&self) -> &DeviceIndex {
        &self.index
    }
}

#[derive(Debug)]
enum DeviceIoEventKind {
    Left,
    Joined(uvc::Device<'static>),
}

impl DeviceIoEventKind {
    #[inline]
    const fn has_left(&self) -> bool {
        match self {
            Self::Left => true,
            Self::Joined(_) => false,
        }
    }

    #[inline]
    const fn has_joined(&self) -> bool {
        !self.has_left()
    }
}

#[derive(Debug)]
struct DeviceIoEvent {
    dev: MinDevice,
    kind: DeviceIoEventKind,
    time: Time,
}

#[inline]
fn is_tis_cam(descriptor: &rusb::DeviceDescriptor) -> bool {
    descriptor.vendor_id() == 0x199e
        && descriptor.product_id() >= 0x8201
        && descriptor.product_id() <= 0x8208
}

impl DeviceIoEvent {
    fn from_min_device(device: MinDevice, kind: DeviceIoEventKind) -> Self {
        Self {
            dev: device,
            kind,
            time: Time::now(),
        }
    }

    #[inline]
    fn has_left(&self) -> bool {
        self.kind.has_left()
    }

    #[inline]
    fn has_joined(&self) -> bool {
        self.kind.has_joined()
    }

    fn idx(&self) -> &DeviceIndex {
        &self.dev.index
    }
}

struct IoThread {
    logging: std::sync::mpsc::Sender<String>,
    device_event: std::sync::mpsc::Sender<DeviceIoEvent>,
    min_event: std::sync::mpsc::Sender<MinIoEvent>,
}

impl<T> rusb::Hotplug<T> for IoThread
where
    T: rusb::UsbContext,
{
    fn device_arrived(&mut self, device: rusb::Device<T>) {
        if let Some(dev) = MinDevice::from_libusb_device(device) {
            refresh_context(uvc::Context::new().unwrap());
            let uvc_dev = get_context()
                .find_device(dev.vendor_id, dev.product_id, None)
                .unwrap();
            let ev = DeviceIoEvent::from_min_device(dev, DeviceIoEventKind::Joined(uvc_dev));
            self.log(&ev);
            self.send_min_event(MinIoEvent {
                index: ev.idx().clone(),
                kind: MinIoEventKind::Joined,
            });
            self.send_event(ev);
        }
    }

    fn device_left(&mut self, device: rusb::Device<T>) {
        self.check(device, DeviceIoEventKind::Left)
    }
}

impl IoThread {
    fn log(&self, dev: &DeviceIoEvent) {
        let formatted_ev = format!("device event: {:?}", dev);
        self.logging.send(formatted_ev).expect("could not log msg");
    }

    fn check<T>(&self, device: rusb::Device<T>, event_kind: DeviceIoEventKind)
    where
        T: rusb::UsbContext,
    {
        if let Some(dev) = MinDevice::from_libusb_device(device) {
            if let Some(min_ev) = MinIoEventKind::from_device_event(&event_kind) {
                self.send_min_event(MinIoEvent {
                    index: dev.idx().clone(),
                    kind: min_ev,
                })
            }
            let event = DeviceIoEvent::from_min_device(dev, event_kind);
            self.log(&event);
            self.send_event(event);
        }
    }

    fn send_event(&self, event: DeviceIoEvent) {
        self.device_event.send(event).expect("could not send event");
    }

    fn send_min_event(&self, ev: MinIoEvent) {
        self.min_event.send(ev).unwrap();
    }

    const fn init(
        logging: std::sync::mpsc::Sender<String>,
        device_event: std::sync::mpsc::Sender<DeviceIoEvent>,
        min_io: std::sync::mpsc::Sender<MinIoEvent>,
    ) -> Self {
        Self {
            logging,
            device_event,
            min_event: min_io,
        }
    }

    fn spin(self) {
        let ctx = rusb::Context::new().unwrap();
        let mut registration = Some(
            rusb::HotplugBuilder::new()
                .enumerate(true)
                .register(ctx.clone(), Box::new(self))
                .unwrap(),
        );

        loop {
            ctx.handle_events(None).unwrap();
            /*
            if let Some(registration) = registration.take() {
                ctx.unregister_callback(registration);
                break;
            }
            */
        }
    }
}

#[repr(transparent)]
#[derive(PartialEq, Clone, Debug)]
pub struct Time(std::time::Duration);

impl Time {
    #[inline]
    fn now() -> Self {
        Time(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap(),
        )
    }

    fn as_millis(self) -> u128 {
        self.0.as_millis()
    }
}
