from conan import ConanFile


class NeoFDTD(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        self.requires("cli11/2.4.2")
        self.requires("fmt/11.0.1")

        if self.settings.os != "Macos":
            self.requires("hdf5/1.14.4.3")
            self.requires("opencv/4.10.0")

    def config_options(self):
        self.options["opencv"].calib3d = False
        self.options["opencv"].dnn = False
        self.options["opencv"].features2d = False
        self.options["opencv"].flann = False
        self.options["opencv"].gapi = False
        self.options["opencv"].highgui = False
        self.options["opencv"].imgcodecs = True
        self.options["opencv"].imgproc = True
        self.options["opencv"].ml = False
        self.options["opencv"].objdetect = False
        self.options["opencv"].photo = False
        self.options["opencv"].stitching = False
        self.options["opencv"].video = False
        self.options["opencv"].videoio = True

        self.options["opencv"].with_eigen = True
        self.options["opencv"].with_openexr = False
        self.options["opencv"].with_png = False
        self.options["opencv"].with_tiff = False
        self.options["opencv"].with_webp = False
        self.options["opencv"].with_ffmpeg = True

        self.options["ffmpeg"].avdevice = False
        self.options["ffmpeg"].avcodec = True
        self.options["ffmpeg"].avformat = True
        self.options["ffmpeg"].swresample = True
        self.options["ffmpeg"].swscale = True
        self.options["ffmpeg"].postproc = True
        self.options["ffmpeg"].avfilter = False
        self.options["ffmpeg"].with_asm = True
        self.options["ffmpeg"].with_zlib = True
        self.options["ffmpeg"].with_bzip2 = True
        self.options["ffmpeg"].with_lzma = True
        self.options["ffmpeg"].with_libiconv = False
        self.options["ffmpeg"].with_freetype = False
        self.options["ffmpeg"].with_openjpeg = False
        self.options["ffmpeg"].with_openh264 = False
        self.options["ffmpeg"].with_opus = False
        self.options["ffmpeg"].with_vorbis = False
        self.options["ffmpeg"].with_libx264 = False
        self.options["ffmpeg"].with_libx265 = False
        self.options["ffmpeg"].with_libvpx = False
        self.options["ffmpeg"].with_libmp3lame = False
        self.options["ffmpeg"].with_libfdk_aac = False
        self.options["ffmpeg"].with_libwebp = False
        self.options["ffmpeg"].with_ssl = False
        self.options["ffmpeg"].with_libalsa = False
        self.options["ffmpeg"].with_pulse = False
        self.options["ffmpeg"].with_vaapi = False
        self.options["ffmpeg"].with_vdpau = False
        self.options["ffmpeg"].with_xcb = False
        self.options["ffmpeg"].with_appkit = True
        self.options["ffmpeg"].with_avfoundation = True
        self.options["ffmpeg"].with_coreimage = True
        self.options["ffmpeg"].with_audiotoolbox = True
        self.options["ffmpeg"].with_videotoolbox = True
        self.options["ffmpeg"].with_programs = False
        self.options["ffmpeg"].with_libsvtav1 = False
        self.options["ffmpeg"].with_libaom = False
        self.options["ffmpeg"].with_libdav1d = False
        self.options["ffmpeg"].with_xlib = False
