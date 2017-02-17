#include "hpc_util.h"
#include "VDeleter.h"

#include <iostream>
#include <stdexcept>//捕获异常
#include <functional>// be used for a lambda functions in the resource management section
#include <chrono>//记录时间
#include <fstream>//文件流
#include <algorithm>//??
#include <vector>
#include <cstring>
#include <array>
#include <set>
#include <unordered_map>  //去除重复的obj顶点数据


const int WIDTH = 800; //glfw  window  size
const int HEIGHT = 600;

const std::string MODEL_PATH = "models/chalet.obj";//加载模型的路径
const std::string TEXTURE_PATH = "textures/chalet.jpg";//获取纹理的路径  纹理图片的尺寸为2^n  一般可以提高性能

const std::vector<const char*> validationLayers = {
	"VK_LAYER_LUNARG_standard_validation"
};//validation layers 通过指定他们的名字来启用

//physics device必须支持如下扩展才能创建swapchain
const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

//VkDebugReportCallbackCreateInfoEXT为扩展的函数，所以 不能自动加载，只能使用vkGetInstanceProcAddr来查找其地址
VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) {
	auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pCallback);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

// VkDebugReportCallbackEXT 对象需要被vkDestroyDebugReportCallbackEXT清除
void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
	if (func != nullptr) {
		func(instance, callback, pAllocator);
	}
}

struct QueueFamilyIndices {
	int graphicsFamily = -1;//图像处理队列
	int presentFamily = -1;//显示队列

	bool isComplete() {
		return graphicsFamily >= 0 && presentFamily >= 0;
	}
};
/*
swapchain必须在vulkan里被显示的创建
//swapchain为待显示的图片队列
swapchain通过刷新屏幕来控制显示图片

Surface 的性能(Capabilities)(比如 : min/max number of images in swap chain, min/max width and height of images)。

Surface 的格式(formats)(比如 : pixel format, color space)。

可用的显示模式(present mode)。


*/
//
struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;//性能
	std::vector<VkSurfaceFormatKHR> formats;//格式
	std::vector<VkPresentModeKHR> presentModes;//显示模式
};

struct Vertex {
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;//如果直接使用纹理坐标当作rgbs值   则g为横坐标  r为纵坐标

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription = {};
		bindingDescription.binding = 0;//指定在binding数组里的索引值
		bindingDescription.stride = sizeof(Vertex);
		/*
		inputRate  有两个值可选   
		VK_VERTEX_INPUT_RATE_VERTEX ： Move to the next data entry after each vertex（每个顶点后）
		VK_VERTEX_INPUT_RATE_INSTANCE  ：Move to the next data entry after each instance（每个实例（一般只在几何着色器中使用，即在顶点着色器中利用实例对象））
		*/
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}
	//一个顶点属性一个VKVertexInputAttributeDescription
	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};
		attributeDescriptions[0].binding = 0;  
		attributeDescriptions[0].location = 0;//索引值
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);//获得texCoord在Vertex里的偏移量

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const {
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}
};

static double startTime, currTime = 0;
static uint32_t nCount = 0;

namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
	}

private:
	GLFWwindow* window;

	VDeleter<VkInstance> instance{ vkDestroyInstance };//vkDestoryInstance方法 clean up the instance
	VDeleter<VkDebugReportCallbackEXT> callback{ instance, DestroyDebugReportCallbackEXT };//添加destory方法
	/*
		 vulkan 平台无关 ，所以使用WSI扩展来连接vulkan和窗口系统
	VK_KHR_surface是一个Instance 级别的扩展，我们在创建Instance时已经通过
	glfwGetRequiredInstanceExtensions允许了这个扩展
	 The window surface needs to be created right after the instance creation, 
	because it can actually influence the physical device selection
	即使vulkan实现支持WSI,但是不意味着每个显卡都支持（指从显卡里寻找一种具有将渲染结果提交(presenting)到surface上的命令的队列(queue family)）
																						   */
	VDeleter<VkSurfaceKHR> surface{ instance, vkDestroySurfaceKHR };

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;//选择显卡来支持我们需要的一些特性  VkPhysicalDevice 将同Instance一同销毁，这里不必使用VDeleter
													 /*
													 显卡Type:
													 VK_PHYSICAL_DEVICE_TYPE_OTHER = 0, //other
													 VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1, //集成
													 VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,  //独立
													 VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3, //虚拟
													 VK_PHYSICAL_DEVICE_TYPE_CPU = 4,  //running on cpu
													 */
	VDeleter<VkDevice> device{ vkDestroyDevice };//用于删除logical deivce  只有physical device是不行的，必须有logical device与其相连  logical device创建与instance类似
	VkQueue graphicsQueue;//图像处理队列（绘制）随Logical Device创建而一起创建、删除
	VkQueue presentQueue;//显示队列

	VDeleter<VkSwapchainKHR> swapChain{ device, vkDestroySwapchainKHR };//声明swapchain
	std::vector<VkImage> swapChainImages;//swapChain里的image随swapChain一起创建、一起销毁
	VkFormat swapChainImageFormat;//image的格式
	VkExtent2D swapChainExtent;//image的分辨率
	//为了使用VkImage,不管是在Swap Chain 还是在Pipeline 中，我们都必须创建VkImageView,
	//就如同它的字面意思一样,imageView是image的一个 view.他描述了我们如何访问image、访问image的哪一部分等
	std::vector<VDeleter<VkImageView>> swapChainImageViews;//创建vkimageView,用作color attachment
	std::vector<VDeleter<VkFramebuffer>> swapChainFramebuffers;

	VDeleter<VkRenderPass> renderPass{ device, vkDestroyRenderPass };
	VDeleter<VkDescriptorSetLayout> descriptorSetLayout{ device, vkDestroyDescriptorSetLayout };
	VDeleter<VkPipelineLayout> pipelineLayout{ device, vkDestroyPipelineLayout };
	VDeleter<VkPipeline> graphicsPipeline{ device, vkDestroyPipeline };

	VDeleter<VkCommandPool> commandPool{ device, vkDestroyCommandPool };

	//depth buffering 用于存储深度值
	VDeleter<VkImage> depthImage{ device, vkDestroyImage };
	VDeleter<VkDeviceMemory> depthImageMemory{ device, vkFreeMemory };
	VDeleter<VkImageView> depthImageView{ device, vkDestroyImageView };

	//create the actual texture image  image and its memory handle
	VDeleter<VkImage> textureImage{ device, vkDestroyImage };
	VDeleter<VkDeviceMemory> textureImageMemory{ device, vkFreeMemory };
	VDeleter<VkImageView> textureImageView{ device, vkDestroyImageView };

	//创建sampler对象  sampler不需要VkImage  可直接从texture提取colors   1D/2D/3D texture都可以使用
	VDeleter<VkSampler> textureSampler{ device, vkDestroySampler };

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	VDeleter<VkBuffer> vertexBuffer{ device, vkDestroyBuffer };
	VDeleter<VkDeviceMemory> vertexBufferMemory{ device, vkFreeMemory };
	VDeleter<VkBuffer> indexBuffer{ device, vkDestroyBuffer };
	VDeleter<VkDeviceMemory> indexBufferMemory{ device, vkFreeMemory };

	VDeleter<VkBuffer> uniformStagingBuffer{ device, vkDestroyBuffer };
	VDeleter<VkDeviceMemory> uniformStagingBufferMemory{ device, vkFreeMemory };
	VDeleter<VkBuffer> uniformBuffer{ device, vkDestroyBuffer };
	VDeleter<VkDeviceMemory> uniformBufferMemory{ device, vkFreeMemory };

	VDeleter<VkDescriptorPool> descriptorPool{ device, vkDestroyDescriptorPool };
	VkDescriptorSet descriptorSet;

	std::vector<VkCommandBuffer> commandBuffers;
	// image 已经得到，可以被渲染了
	VDeleter<VkSemaphore> imageAvailableSemaphore{ device, vkDestroySemaphore };
	// image 渲染完毕可以被提交显示了
	VDeleter<VkSemaphore> renderFinishedSemaphore{ device, vkDestroySemaphore };



	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

		glfwSetWindowUserPointer(window, this);
		glfwSetWindowSizeCallback(window, HelloTriangleApplication::onWindowResized);
	}

	void initVulkan() {
		createInstance();
		setupDebugCallback();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createDepthResources();
		createFramebuffers();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		loadModel();//obj模型加载
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffer();
		createDescriptorPool();
		createDescriptorSet();
		createCommandBuffers();
		createSemaphores();
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();

			updateUniformBuffer();
			calculate();
			drawFrame();
		}

		vkDeviceWaitIdle(device);// 等待具体某个命令队列里的的某个操作的结束
	}
	
	static void calculate()
	{
		//auto currentTime = std::chrono::high_resolution_clock::now();
		//float time = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() / 1000.0f;


		/*std::cout << "startTime ：" << startTime << std::endl;
		std::cout << "currTime ：" << currTime << std::endl;
		*/
		if (nCount == 0)
		{
			nCount++;
			startTime = glfwGetTime();
		}
		else if ((currTime - startTime) > 1.0)
		{
			std::cout << "帧速率：" << nCount << std::endl;
			nCount = 0;
		}
		else
		{
			currTime = glfwGetTime();
			nCount++;
		}
	}

	static void onWindowResized(GLFWwindow* window, int width, int height) {
		if (width == 0 || height == 0) return;

		HelloTriangleApplication* app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->recreateSwapChain();
	}

	void recreateSwapChain() {
		vkDeviceWaitIdle(device);

		createSwapChain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createDepthResources();
		createFramebuffers();
		createCommandBuffers();
	}
	/*
	验证层做的工作

	检查参数是否被勿用。
	追踪对象的创建与销毁，检测是否与有资源泄露。
	追踪线程(thread)调用的源头，检测线程是否安全。
	将方法调用的参数打印到标准输出。
	Tracing Vulkan calls for profiling and replaying。

	可以免费堆放这些验证层包括所有你感兴趣的调试功能。
	*/
	
	//创建vkinstance
	void createInstance() {
		//进行验证  在发布的代码中可以不包括validationLayer
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}
		//关于应用的信息
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;//必须
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);//必须
		appInfo.apiVersion = VK_API_VERSION_1_0;//必须

		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = extensions.size();
		createInfo.ppEnabledExtensionNames = extensions.data();

		if (enableValidationLayers) {//有可支持的validation layer
			createInfo.enabledLayerCount = validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}
		//replace 方法 调用clean up for any existing handle and then gives you a non-const pointer to overwrite the handle
		//instance将会存储到VKInstance的成员里
		if (vkCreateInstance(&createInfo, nullptr, instance.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}
	}
	/*
	在create function里  eg:vkCreateInstance
	Pointer to struct with creation info
	Pointer to custom allocator callbacks, always nullptr in this tutorial
	Pointer to the variable that stores the handle to the new object
	*/

	void setupDebugCallback() {
		if (!enableValidationLayers) return;

		VkDebugReportCallbackCreateInfoEXT createInfo = {};//为callback创建描述信息   
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;//用于debug  error、warning
		createInfo.pfnCallback = debugCallback;//pUserData 也可以指定数据，类似userData

		if (CreateDebugReportCallbackEXT(instance, &createInfo, nullptr, callback.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug callback!");
		}
	}

	//创建surface  关于渲染和显示
	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, surface.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}
	//选择物理设备（显卡）
	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());//枚举出当前设备可用的所有显卡

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}
	//创建Logical Device
	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);//表示获取的合适的显卡的索引值

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;//设备队列创建描述信息
		std::set<int> uniqueQueueFamilies = { indices.graphicsFamily, indices.presentFamily };//绘制和显示队列

		float queuePriority = 1.0f;
		for (int queueFamily : uniqueQueueFamilies) {
		std::cout << "queueFamily" << uniqueQueueFamilies.size() << std::endl;

			VkDeviceQueueCreateInfo queueCreateInfo = {};//创建queueCount个queueFamilyIndex类型的队列
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; 
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;//目前只需要一个queue family
			queueCreateInfo.pQueuePriorities = &queuePriority;//队列的优先级 0-1
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures = {};  //特性采用默认值

		VkDeviceCreateInfo createInfo = {};//对队列(queue)和特性(features)支持的限定外，还有对Validation layers 和 Extensions的限定，
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos.size();

		createInfo.pEnabledFeatures = &deviceFeatures; 

		//enableValidationLayers和validationLayers直接取自创建VkInstances时已有的定义
		createInfo.enabledExtensionCount = deviceExtensions.size();
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		//创建logical device
		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, device.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		/*
		device : logical device.
		indices.graphicsFamily : 队列种类。
		queueIndex : 这里是 0 ，因为只创建了一个队列，所以这里索引为0.
		VkQueue * : &graphicsQueue。
		*/
		vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);//获取队列 handle
		vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
	}

	//创建交换链
	/*
	它能为我们提供要渲染的图片，然后渲染结的果可以显到屏幕上。
	Swap Chain必须被Vulkan显示的创建。从本质上讲，Swap Chain就是一个图片的队列(a queue of images),这里的图片等着被显示到屏幕上。
	我们的应用将会获得一个图片，然后绘画它，之后将它提交到队列中去。
	Swap Chain 通常的作用是通过屏幕刷新率(refresh rate of the screen)来同步控制图片的显示。

	为了寻找最佳的Swap Chain设置，我们决定从以下三个方面入手:

    Surface (格式)format (如:color depth)
    显示模式(Presentation mode)(如:渲染后的图片“交换”到显示器的时机).
    交换的大小(Swap extent)(如:图片在swap chain里的分辨率)

	*/
	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);//获得支持的细节

		//寻找最佳的swapchain设置
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);//get format
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);//get presentmode
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);//get extent  image的分辨率
	//	std::cout << "imageCount:" << swapChainSupport.capabilities.minImageCount << std::endl;
		//minImageCount 个image 已经十分合适了，但是为了更好的支持三缓冲，我们又多加了一个。maxImageCount 如果为0， 表示不对最大数量做任何限制。

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;//设置Swap Chain 中image的数量，本质上是指队列的长度:
		std::cout << "imageCount:" << imageCount << std::endl;

		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {//该maxImageCount =0
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		//std::cout << "imageCount:" << swapChainSupport.capabilities.maxImageCount << std::endl;

		VkSwapchainCreateInfoKHR createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		createInfo.minImageCount = imageCount;//???
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;//image的层次  一般为1 除非创建3D应用
		/*
		imageUsage：
		如果你想先渲染一个单独的图片然后再进行处理，那就应该使用VK_IMAGE_USAGE_TRANSFER_DST_BIT并使用内存转换操作将渲染好的image 转换到SwapChain里。
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT 表示image可以用作创建VkImageView，在VkFrameBuffer中适合使用color 或者 reslove attachment.*/
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;//表示用swapchain做什么操作  这里主要是对image进行渲染，image被当作颜色附件来处理

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { (uint32_t)indices.graphicsFamily, (uint32_t)indices.presentFamily };

		if (indices.graphicsFamily != indices.presentFamily) {
			/*
			如果grapics queue 和 present queue不相同，就会出现这多种队列访问image的情况：
			我们在grapics queue 中绘画image,然后将它提交到presention queue 去等待显示
			imageSharingMode取值：
			VK_SHARING_MODE_EXCLUSIVE : image 一段时间内只能属于一种队列，所有权的转换必须明确声明，这个选项可以提供较好的性能。
			VK_SHARING_MODE_CONCURRENT : image 可以跨多种队列使用，所有权的转换不必明确声明。
			*/
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;//imageSharingMode 表示多种队列中，image如何使用
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;//不对Image  变换
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;// 忽略和其他窗口颜色混合时的Alpha 通道
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;// 不处理那些被遮盖的像素 

		VkSwapchainKHR oldSwapChain = swapChain;
		createInfo.oldSwapchain = oldSwapChain;

		VkSwapchainKHR newSwapChain;
		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &newSwapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}
		swapChain = newSwapChain;
		//获得swapchain里的所有image
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		//std::cout << "imageCount:" <<imageCount << std::endl;

		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;//设置format格式
		swapChainExtent = extent;//设置extent
	}

	//为swap chain里的image设置imageview
	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size(), VDeleter<VkImageView>{device, vkDestroyImageView});

		for (uint32_t i = 0; i < swapChainImages.size(); i++) {//为每个image创建imageview
			createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, swapChainImageViews[i]);
		}
	}

	/*
	创建Render Pass 需要Subpass 和Attachment, 现在就简单的理解为Render Pass 包含Subpass 数组和 
	Attachment数组吧。Render Pass的工作需要Subpass 来完成， 每一个Subpass 可以操作多个Attachment ,
	怎么从Attachment数组中表示哪些attachment会被某个Subpass处理呢，
	所以我们需要一个VkAttachmentReference来描述attachment在Attachment数组中的下标和处理时的layout。
	*/

	/*
	在创建Pipeline 之前我们必须告诉Vulkan在渲染时要使用的FrameBuffer 附件(attachments)，需要定义使用color buffer 以及
	depth buffer attachments的数量，
	要使用多少个采样(samples)以及应该如何处理采样的内容。所有这些信息都可以填写在Render Pass里
	*/
	void createRenderPass() {
		VkAttachmentDescription colorAttachment = {};
		colorAttachment.format = swapChainImageFormat;//颜色附件格式必须与swapchain里的一个image匹配
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;//不使用抗锯齿，所以为1

		// subpass 执行前对color或depth attachment内容做何种处理
		/*loadOp和storeO表示渲染前和渲染后要做的动作，在我们的例子中,写入新片原(fragment)之前先清空FrameBuffer,使FrameBuffer变为黑色。
		我们想让渲染后的三角形显示到屏幕上，所以这里我们将storeOp 设置为保存。*/
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		/*
		
		VK_ATTACHMENT_STORE_OP_STORE: Rendered contents will be stored in memory and can be read later
		VK_ATTACHMENT_STORE_OP_DONT_CARE: Contents of the framebuffer will be undefined after the rendering operation

		*/
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		/*
		在Vulkan中，用具有特定像素格式的VkImage 表示纹理(texture)和FrameBuffer,而像素在内存中的布局(layout)随着我们使用Image 
		的目的不同是可以改变的。一些常见的layout有:

		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: Images 用作 color attachment
		VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: 表示一个要被显示的swap chain image源
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: images 作为内存拷贝操作的目的
		*/
		/*
		initial / finalLaout表示渲染前后image的layout, VK_IMAGE_LAYOUT_UNDEFINED表示我们不在乎之前的layout,
		而VK_IMAGE_LAYOUT_PRESENT_SRC_KHR表示在渲染后，使用Swpa Chain 时，image 处于可显示的layout
		*/
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentDescription depthAttachment = {};//depth attachment
		depthAttachment.format = findDepthFormat();//找到合适的format  必须与 depth image 相同
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;//don't care now   允许硬件执行额外的操作
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		//如果在render过程中 The layout of the image没有变化，则initial和final相同
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		//每一个Subpass 引用一个或多个我们在前一节用VkAttachmentDescription 描述的attachment(s) ,每一个引用用VkAttachmentReference描述:
		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;//值是一个索引(index)  表示代表哪个附件
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef = {};//此为subpass添加的深度附件
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		/*
		一个Render Pass 由一系列的subPass组成，并由subpass 处理随后的渲染操作,代表渲染的一个阶段,
		渲染命令存储在一个Render Pass的诸多subpass中，一个subpass的动作取决于上一个subpass 的处理结果，
		如果我们把它们打包成一个Render Pass ,Vulkan 能够为我们重新排序它们的执行顺序，节省内存带宽，从而可能获取更好的性能
		*/
		//pipelineBindPoint表示要绑定的Pipeline类型，这里只支持两种类型:计算(compute)和图形(graphics)，这里我们选择图形Pipeline
		VkSubpassDescription subPass = {};
		subPass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subPass.colorAttachmentCount = 1;
		subPass.pColorAttachments = &colorAttachmentRef;
		subPass.pDepthStencilAttachment = &depthAttachmentRef;
		/*
		
		Subpass 还可以引用的attachment有:

		pInputAttachments: 从着色器中读取的 attachment
		pResolveAttachments: multiple color attachment
		pDepthStencilAttachment: depth and stencil data 的attachment
		pPreserveAttachments: 不被Subpass 使用，但出于某种原因需要保存

		*/


		/*
		ender Pass 中的 subpass 自动处理image (attachment)的layout转换，
		这些转换被subpass 依赖(subpass dependencies) 所控制,它指定了subpass间的内存和执行依赖，
		虽然我们只有一个subpass,但是在执行这个subpass的前后操作也被隐式的当做subpass了。

		有两个内置的依赖(built-in dependencies)控制render pass前和render pass后的转换，
		前者出现的时机并不正确，因为它假定render pass前的转换发生在pipeline开始的时候，
		但是这个时候我们还没有获得image呢！因为image是在VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT阶段(stage)才获得的。
		所以我们必须重写/覆盖这个依赖。
		*/
		VkSubpassDependency dependency = {};
		/*
		srcSubpass和dstSubpass分别表示依赖的索引和从属的subpass(即生产者与消费者的索引).
		VK_SUBPASS_EXTERNAL代表Render pass 前或后的隐含的subpass,
		这取决于VK_SUBPASS_EXTERNA是被定义在src还是dst。
		索引0指向我们定义的第一个同时也是唯一的一个subpass。
		dst必须大于src 以防止循环依赖。
		*/
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		/*
		以下这两个字段分别定义了我们将在什么操作上等待以及这个操作在何种阶段发生。
		我们必须等待Swap Chain从image读完之后才能访问它，这个操作发生在pipeline的最后阶段。
		*/
		dependency.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependency.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		/*
		以下两个参数表示读和写 color attachment 操作必须在 _COLOR_ATTACHMENT_ 阶段进行等待，
		这些设置保证:除非必要(如：当我们真的想写颜色(color)的时候)否则转换将不会发生。
		*/
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		/*
		Attachment 和 Subpass 都已经声明好了，现在开始创建Render Pass：
		*/

		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };//具有颜色和深度两个附件
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = attachments.size();
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subPass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, renderPass.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void createDescriptorSetLayout() {
		/*
		
		创建descriptor需要   descriptor layout, descriptor pool and descriptor set 
		*/
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		//create a new descriptor "combined image sampler" 保证shader可以接受image通过sampler
		VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;//stageFlags表示 combined image sampler descriptor 将会在shader使用

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = 
		{ uboLayoutBinding, samplerLayoutBinding };
		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = bindings.size();
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, descriptorSetLayout.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}
	
	void createGraphicsPipeline() {
		auto vertShaderCode = readFile("shaders/vert.spv");//获得着色器的二进制文件
		auto fragShaderCode = readFile("shaders/frag.spv");

		VDeleter<VkShaderModule> vertShaderModule{ device, vkDestroyShaderModule };
		VDeleter<VkShaderModule> fragShaderModule{ device, vkDestroyShaderModule };
		createShaderModule(vertShaderCode, vertShaderModule);
		createShaderModule(fragShaderCode, fragShaderModule);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		//指明使用顶点着色器，并从main函数开始调用
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};//VkPipelineVertexInputStateCreateInfo 表示传递顶点的格式
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		/*

		顶点数据的描述(Bindings) :数据间的间隔，
		以及判断数据是顶点数据(pre-vertex)还是实例数据(pre-instance)。

		顶点属性的描述（Attribute Descriptions)：传入到Vertex Shader 
		里的属性(attributes)类型，从哪个Binding加载以及offset。
		
		*/
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		//画什么样的几何图形和图元顶点是否可以重用
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		/*
		topology  类型：
		VK_PRIMITIVE_TOPOLOGY_POINT_LIST: 画点
		VK_PRIMITIVE_TOPOLOGY_LINE_LIST: 每两个点为一条线，顶点不能重用
		VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: 一条线的第二个顶点可以作为下一条线的起点(可重用)
		VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: 每三个点一个三角形，顶点不可重用
		VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: 一个三角形的第三个点可以作为下一个三角形的起点(可重用)

		*/
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;//图元是否重启  与_STRIP topology modes 有关

		//Viewport 其实就是输出结果被渲染到FrameBuffer的多大区域中。它总是从坐标(0,0)点开始,
		//具有一定宽(width)和高(height)的矩形区域。这个区域我们用VkViewport表示
		/*
		正如在创建swpaChain时所描述的那样，swapChain及其Image的尺寸可能和window的尺寸不同。我们用Swap Chain的 width和height 赋值Viewport, 
		因为接下来Swap Chain的 images 将作为FrameBuffer使用。Min/maxDepth表示FrameBuffer的深度范围，深度取值在[0.0 , 1.0]范围内，
		注意，minDepth可能大于maxDepth,如果没有什么特殊需要，我们将按照标准的定义， 即:minDepth=0 , maxDepth=1.0
		*/
		//Viewport 定义了image 到 FrameBuffer的变换，而Scissor 矩形框决定哪些区域的像素将会被存储，
		//在Scissor矩形框外的像素将会在光栅化(像素化)阶段被丢弃。所以，比起变换，Scissor 更像是一个过滤器。
		VkViewport viewport = {};//设置视口
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};//创建剪裁窗口
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		/*
		注意,从VkPipelineViewportStateCreateInfo的结构上来看，在某些显卡上，我们可以使用多个Viewport和多个Scissor ，
		这涉及到显卡的支持，在我们创建Logicsl Device时，
		VkPhysicalDeviceFeatures 字段里有 VkBool32 multiViewport;的定义，你可以检查自己的显卡是否支持这个特性。
		*/
		VkPipelineViewportStateCreateInfo viewportState = {};//将视口与剪裁窗口结合起来
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		/*
		光栅化(像素化)，它把来自Vertex Shader 操作后顶点组成的几何图形离散化成一个个片原(fragment)，
		然后将片原传递到Fragment Shader 里进行着色。
		光栅化也执行depth testing、face culling 和 scissor test。你可以配置，
		选择是将整个多边形离散化成片原，还是只离散化边框(edges)
		(又叫 : wireframe rending),我们通过如下结构来进行设置
		*/
		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		/*
		//如果设置成VK_TRUE,那些在视景体近平面(near)和远平面(far)之外的片原将会被拉紧/截取
		现在为false则表示为丢弃
		*/
		rasterizer.depthClampEnable = VK_FALSE;
		/*
		rasterizerDiscardEnable 如果为VK_TRUE, 
		几何数据(geometry)将无法通过Rasterization阶段，
		FrameBuffer 将得不到任何输出数据
		*/
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		/*
		polygonMode取值：

		VK_POLYGON_MODE_FILL: 填充整个多边形区域的片原
		VK_POLYGON_MODE_LINE: 只有多边形边界(edges)的片原
		VK_POLYGON_MODE_POINT: 只画多边形顶点
		*/
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;//线条粗细
		/*  卷绕方式与背面裁剪
			规定裁剪那个面:前面和背面，从摄像机的角度看，顶点按逆时针组成的图形
			是正面，顺时背面。正反面的卷绕方式可以自定义。
		*/
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;//表示裁剪方式 
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling = {};//多重采样可以执行防锯齿的功能
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		//深度测试功能需通过VkPipelineDepthStencilStateCreateInfo开启
		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		//下面两个参数表示新片元需被比较和写入
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;//该参数在绘制透明物体时比较有用
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		//用于深度范围的测试？？
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		/*
		混合(mix) 新颜色和旧颜色来产生最终的颜色。

		将新颜色和旧颜色通过按位运算(bitwise)整合在一起。

		*/
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | 
			VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		//false  为alpha通道混合
		colorBlending.logicOpEnable = VK_FALSE;//按位混合(bitewise combination)， logicOpEnable就要设置成VK_TRUE  
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		VkDescriptorSetLayout setLayouts[] = { descriptorSetLayout };
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = setLayouts;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, pipelineLayout.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;//index
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;//只有一个pipeline

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, graphicsPipeline.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}
	}
	/*
	The color attachment differs for every swap chain image, 
	but the same depth image can be used by all of them 
	because only a single subpass is running at the same time due to our semaphores
	*/
	/*
	在创建Render Pass时,我们期望拥有一个和Swap Chain 里image具有相同格式(format)的FrameBuffer

	将attachments包裹在FrameBuffer中，FrameBuffer 通过引用VkImageView来关联所有的attachments

	在本案例中只有一个attachment : color attachment 。然而作为attachment的image取决于在显示的时候Swap Chain
	到底返回的是哪一个image，这就意味着我们需要为Swap Chain里的每一个image 创建一个FrameBuffer
	*/
	void createFramebuffers() {

		//std::cout << "swapChainImageViews.size() :" << swapChainImageViews.size() << std::endl;
		swapChainFramebuffers.resize(swapChainImageViews.size(), VDeleter<VkFramebuffer>{device, vkDestroyFramebuffer});

		// 遍历每一个imageView (image) 为它创建Framebuffer
		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			std::array<VkImageView, 2> attachments = {//有两个附件
				swapChainImageViews[i],
				depthImageView
			};

			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			// 这里的 attachmentCount 与pAttachments 和Render Pass 里的
			// pAttachment  相关联
			framebufferInfo.attachmentCount = attachments.size();
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;//swap chain image只是一个

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, swapChainFramebuffers[i].replace()) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}
	/*
	在Vulkan中，像绘画命令、内存转换等操作并不是直接通过方法调用去完成的，而是需要把所有的
	操作放在Command Buffer 里。
	这样的一个好处就是：那些已设置好的具有难度的绘图工作都可以在多线程的情况下提前完成。
	*/
	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		//Command pools 管理Command buffer 的内存而且Command buffer 从Command pool中被创建
		VkCommandPoolCreateInfo poolInfo = {};//首先创建commandpool
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		/*
		command buffer 需要提交到队列中等待执行，像我们已经得到的图形队列(graphics)和
		显示队列(presentation)。
		从一个Command pool 产生所有command buffers 只能对应一种特定的队列。
		因为我们想使用绘图命令，所以选择graphics 队列 。
		*/
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;//绘制图像队列

		/*

		flags的取值只有两个:

		VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: Command buffer 比较短命，可能会在一个相对较短
		的时间内被重置或释放，主要用于控制pool中内存的分配行为。
		VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: commad buffer是否可以被分别重置，
		如果无此参数，pool中所有的command buffer都将被重置。

		只是在程序的开始时记录Command buffer ,然后在main loop 中调用多次，所以不需要flags参数
		*/

		if (vkCreateCommandPool(device, &poolInfo, nullptr, commandPool.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics command pool!");
		}
	}
	//需创建Image ImageView 以及transitionImageLayout
	void createDepthResources() {
		VkFormat depthFormat = findDepthFormat();

		createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, 
			VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
		createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 
			depthImageView);

		//只有转换从layout才适合depth attachment 使用
		transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, 
			VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	}

	/*
	 VkFormatProperties可取以下三个值

	 linearTilingFeatures: Use cases that are supported with linear tiling
	 optimalTilingFeatures: Use cases that are supported with optimal tiling
	 bufferFeatures: Use cases that are supported for buffers

	*/
	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}

	/*
	
    VK_FORMAT_D32_SFLOAT: 32-bit float for depth
    VK_FORMAT_D32_SFLOAT_S8_UINT: 32-bit signed float for depth and 8 bit stencil component
    VK_FORMAT_D24_UNORM_S8_UINT: 24-bit float for depth and 8 bit stencil component

	*/
	VkFormat findDepthFormat() {
		return findSupportedFormat(
		{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
			);
	}
	//查看depth format 是否含有stencil component
	bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	//加载image 并output texture
	void createTextureImage() {
		int texWidth, texHeight, texChannels;//纹理的宽高度  及  通道
		/*
		load image 
		STBI_rgb_alpha  为stb_image.h中定义的  一般在加载纹理时会要求有一个透明度，视纹理类型来定，
			为保证和将来的其他纹理保持一致，jpg为3通道
		输出  纹理的宽高  、通道数量
		stbi_uc  为无符号char 型数据
		pixels指针返回的是char array里的首元素的值
		*/

		stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = texWidth * texHeight * 4;//4 bytes per pixel

	//	std::cout << "texture size " << sizeof(char) << std::endl;

		if (!pixels) {//如果加载失败，
			throw std::runtime_error("failed to load texture image!");
		}

		VDeleter<VkImage> stagingImage{ device, vkDestroyImage };
		VDeleter<VkDeviceMemory> stagingImageMemory{ device, vkFreeMemory };
		//为线性 LINEAR 平铺展开   采用最常用的格式R8G8B8A8
		createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingImage, stagingImageMemory);
		/*
		the VK_FORMAT_R8G8B8A8_UNORM format is not supported by the graphics hardware
		*/
		//该内存需被主机可见（host visible ）使用vkMapMemory
		void* data;
		vkMapMemory(device, stagingImageMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, (size_t)imageSize);
		vkUnmapMemory(device, stagingImageMemory);
		//当image data 复制进memory后，需clear pixel array
		stbi_image_free(pixels);

		/*final image 纹理尺寸与stagingImage一样，
		format也应该兼容stagingImage（因为仅仅只是copy 原始 image data 的命令）
		tiling mode 不需要一样
		该纹理对象被作为转换的目标图像 ，并且希望在着色器中可以对其纹理数据进行采样
		为获得更好的性能，memory应为device local
		*/
		createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | 
			VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

		/*
		将stage image 复制给 textureImage 

		Transition the staging image to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
		Transition the texture image to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
		Execute the image copy operation

		*/
		//VK_IMAGE_LAYOUT_PREINITIALIZED and VK_IMAGE_LAYOUT_UNDEFINED  都可以用于 oldLayout 当transition image
		transitionImageLayout(stagingImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyImage(stagingImage, textureImage, texWidth, texHeight);
		//保证可以在shader里对textureImage进行采样
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}

	void createTextureImageView() {
		createImageView(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, textureImageView);
	}
	/*
	
	address mode 类型
	
    VK_SAMPLER_ADDRESS_MODE_REPEAT: Repeat the texture when going beyond the image dimensions.
    VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: Like repeat, but inverts the coordinates to mirror the image when going beyond the dimensions.
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: Take the color of the edge closest to the coordinate beyond the image dimensions.
    VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: Like clamp to edge, but instead uses the edge opposite to the closest edge.
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: Return a solid color when sampling beyond the dimensions of the image.

	
	*/
	void createTextureSampler() {
		//sampler 通过VkSamplerCreateInfo来配置
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		//mag 和min采样  mag  着重 oversampling   min 着重 undersampling
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;//各异项性过滤  一般都开启  除非性能很差
		samplerInfo.maxAnisotropy = 16;//可用于计算最终颜色的采样值个数   值越小性能越好但是质量差   目前没有超过16的硬件支持
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;//clamp to border 当采样超出image size时，填充 black, white or transparent
		/*;//指定texture的坐标系  
		为true 时  within the [0, texWidth) and [0, texHeight) range

		false   [0, 1) range on all axes
		*/
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;//如果开启  用于与一个值进行比较，结果将用于filtering    一般用于shadow map
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;//用于mipmap

		if (vkCreateSampler(device, &samplerInfo, nullptr, textureSampler.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	void createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, VDeleter<VkImageView>& imageView) {
		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;//viewType 有VK_IMAGE_VIEW_TYPE_1/2/3D
		viewInfo.format = format;
		//Component 字段 采用默认值

		//subresourceRange描述image的使用目的和要被访问的部分
		viewInfo.subresourceRange.aspectMask = aspectFlags;//aspectFlags  取值为 VK_IMAGE_ASPECT_COLOR_BIT
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(device, &viewInfo, nullptr, imageView.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}
	}

	//the width, height, format, tiling mode, usage, and memory properties parameters,
	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, 
		VkImageUsageFlags usage,
			VkMemoryPropertyFlags properties, VDeleter<VkImage>& image, VDeleter<VkDeviceMemory>& imageMemory) {
		//图像的参数
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		/*一维的图像可以用来存储数据或渐变数组
		二维的一般用于图像
		三维的可以用于存储体积像素值
		*/
		imageInfo.imageType = VK_IMAGE_TYPE_2D;//create 1D, 2D and 3D images
		imageInfo.extent.width = width; //extern 字段用于指定图像的尺寸（各个轴）
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		//当前的纹理不是多维数组，也不使用mipmapping  则为1  
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		
		imageInfo.format = format;//格式
		 //如果想要在image的memory中直接获取纹理元素，则采用Linear  tiling
		imageInfo.tiling = tiling;//展开方式  
		/*初始化layout有两种选择，
		一种是undefined  在第一次transition(变换)时，就会discard texels
		一种是PREINITIALIZED  在第一次变换时，不会discard texels
		undefiend 适合image作为颜色或深度缓冲的附件时使用  因为在renderpass之前会将数据clear
		如果想要对texture填充数据，则需使用preinitialized
		*/
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
		//usage:因为stagingImage将会复制给finalImage,所以类型为transfer_src
		imageInfo.usage = usage;

		//samples与多重采样有关  只需要图像作为附件，所以只要一个sample
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.flags = 0; //flags与sparse image 有关，在使用3dtexture时，可采用Sparse images来避免浪费内存
		//该image只被一个支持transfer操作的queue family 使用，
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(device, &imageInfo, nullptr, image.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		//分配内存给image  
		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, imageMemory.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}
	//处理layout translate
	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();
		//using an image memory barrier 来提高layout transition
		//pipeline barrier  通常被用于对资源的同步访问   保证在向一个buffer读取数据时写入操作已完成
		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		//如果不care image存在的内容，oldLayout 则可以使用undefined
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		//一般给定queue family 的index，如果没有给定，则ignored即可
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//指定被作用的image   以及 具有特殊部分的image
		barrier.image = image;
		
		//即使不使用stencil attachment ，也要将其加入到depth_image的 layout transitions
		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

			if (hasStencilComponent(format)) {
				barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			}
		}
		else {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		//image既不是一个数组也不是mipmapping 所以level和layer都为1
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		//barrier主要是用于同步的。
		/*
		必须指定哪些涉及到资源的操作发生在barrier之前，哪些必须wait on barrier
		srcAccessMask  dstAccessMask  值取决于old and new layout,
		*/
		/*
		三种转换需要处理  以后需要其他转换，直接在下面加入即可

		Preinitialized → transfer source: transfer reads should wait on host writes
		Preinitialized → transfer destination: transfer writes should wait on host writes
		Transfer destination → shader reading: shader reads should wait on transfer writes
		*/
		if (oldLayout == VK_IMAGE_LAYOUT_PREINITIALIZED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_PREINITIALIZED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			//只有到这一步才可以说明 image 可以作为depth的附件使用了
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		//所有的pipeline barrier 都是用相同的方法来提交的
		/*
		@commandBuffer //第一个参数指定在pipeline阶段，在barrier之前，应该发生的操作
						  
		@(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, )
				上述参数指定在pipeline阶段哪种操作将会 wait on barrier,并且我们希望该操作立即发生，
			所以在pipeline的top位置就发生，src 和dst相同
		@0		//第三个参数可为0/VK_DEPENDENCY_BY_REGION_BIT.  第二个值表示将barrier转换成per-region 状态
			eg:这意味着实现已经被允许可以开始向到目前为止写入的部分的资源进行读取
		最后三组参数表示pipeline barrier 三种可用的数组 (memory barriers、buffer memory barriers 、
		image memory barriers)	
		*/

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
			
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
			);

		endSingleTimeCommands(commandBuffer);
	}
	
	void copyImage(VkImage srcImage, VkImage dstImage, uint32_t width, uint32_t height) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageSubresourceLayers subResource = {};
		subResource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subResource.baseArrayLayer = 0;
		subResource.mipLevel = 0;
		subResource.layerCount = 1;

		//指定哪一部分的image需要被复制给另外的image的哪一部分
		VkImageCopy region = {};
		region.srcSubresource = subResource;
		region.dstSubresource = subResource;
		region.srcOffset = { 0, 0, 0 };
		region.dstOffset = { 0, 0, 0 };
		region.extent.width = width;
		region.extent.height = height;
		region.extent.depth = 1;
		//image copy操作通过使用vkCmdCopyImage来入队列
		/*
		前两组参数指定src/dst 的image/layout
		*/
		vkCmdCopyImage(
			commandBuffer,
			srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &region
			);

		endSingleTimeCommands(commandBuffer);
	}

	void loadModel() {
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, MODEL_PATH.c_str())) {
			throw std::runtime_error(err);
		}

		std::unordered_map<Vertex, int> uniqueVertices = {};

		for (const auto& shape : shapes) {
			for (const auto& index : shape.mesh.indices) {
				Vertex vertex = {};

				vertex.pos = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};
				vertex.color = {
					1,0,0,
				};

				vertex.texCoord = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};

				if (uniqueVertices.count(vertex) == 0) {
					uniqueVertices[vertex] = vertices.size();
					vertices.push_back(vertex);
				}
				indices.push_back(uniqueVertices[vertex]);
			}
		}
	}

	void createVertexBuffer() {
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		VDeleter<VkBuffer> stagingBuffer{ device, vkDestroyBuffer };
		VDeleter<VkDeviceMemory> stagingBufferMemory{ device, vkFreeMemory };
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
	}

	void createIndexBuffer() {
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VDeleter<VkBuffer> stagingBuffer{ device, vkDestroyBuffer };
		VDeleter<VkDeviceMemory> stagingBufferMemory{ device, vkFreeMemory };
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);
	}
	/*
	使用descriptor包括如下三方面：	

	Specify a descriptor layout during pipeline creation
	Allocate a descriptor set from a descriptor pool
	Bind the descriptor set during rendering


	*/

	void createUniformBuffer() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);
		//VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: the memory should be mapped so that the CPU (host) can access it.
		/*VK_MEMORY_PROPERTY_HOST_COHERENT_BIT :requests that the writes to the memory by the host are visible
			to the device (and vice-versa) without the need to flush memory caches
		*/
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformStagingBuffer, uniformStagingBufferMemory);
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, uniformBuffer, uniformBufferMemory);
	}

	void createDescriptorPool() {
		std::array<VkDescriptorPoolSize, 2> poolSizes = {};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = 1;
		
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;//image sampler
		poolSizes[1].descriptorCount = 1;

		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = poolSizes.size();
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = 1;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, descriptorPool.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void createDescriptorSet() {
		VkDescriptorSetLayout layouts[] = { descriptorSetLayout };
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = layouts;

		if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor set!");
		}

		VkDescriptorBufferInfo bufferInfo = {};
		bufferInfo.buffer = uniformBuffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(UniformBufferObject);

		//allocate a descriptor set with this layout
		// bind the actual image and sampler resources to the descriptor in the descriptor set.
		VkDescriptorImageInfo imageInfo = {};
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo.imageView = textureImageView;
		imageInfo.sampler = textureSampler;

		std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = descriptorSet;
		descriptorWrites[0].dstBinding = 0;//绑定的索引值  为0
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &bufferInfo;//使用bufferInfo

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = descriptorSet;
		descriptorWrites[1].dstBinding = 1;//绑定的索引值  为1
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pImageInfo = &imageInfo;//使用imageInfo
		//到此为止，可以在shader里使用  sampler  

		vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VDeleter<VkBuffer>& buffer, VDeleter<VkDeviceMemory>& bufferMemory) {
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, buffer.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, bufferMemory.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion = {};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void createCommandBuffers() {
		if (commandBuffers.size() > 0) {
			vkFreeCommandBuffers(device, commandPool, commandBuffers.size(), commandBuffers.data());
		}
		/*
		因为绘画命令涉及绑定到正确的VkFrameBuffer,
		所以我们要为Swap Chain里的每一个image 创建一个Command buffer：
		*/
		commandBuffers.resize(swapChainFramebuffers.size());

		//std::cout << "commandBuffers:" << commandBuffers.size() << std::endl;

		VkCommandBufferAllocateInfo allocInfo = {};//分配command buffer
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		/*
		level 字段限定了command buffer 是主要的还是次要的，它有如下取值:
		VK_COMMAND_BUFFER_LEVEL_PRIMARY: 可以提交到队列中执行，但不能从其他command buffer 中调用。
		VK_COMMAND_BUFFER_LEVEL_SECONDARY: 不能直接提交到队列，但可以从主command buffer 中调用。 
		
		此外CommandBuffer 的清理工作和之前的对其他对象的清理工作不同，使用:vkFreeCommandBuffers(),
		它接收一个Command Pool和一个CommandBuffer数组
		*/
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}

		for (size_t i = 0; i < commandBuffers.size(); i++) {

			//通过vkBeginCommandBuffer来开始记录 Command buffer, 并用VkCommandBufferBeginInfo来描述command buffer的具体用法 :
			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			/*
			flags 定义我们该如何使用command buffer ,可能取值:

			VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: 只能被提交一次，之后可能被重置。
			VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: 整个次command buffer 将会在reder pass中，
			主command buffer 将忽略此值。
			VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: command buffer 在等待执行时可以被重复提交

			在这里我们使用_SIMUTANEOUS_USE_BIT，因为很有可能在上一个帧(frame)尚未画完，
			下一个帧的绘画请求就已经提交了。
			pInheritanceInfo表示次command buffer 从主command buffer 继承过来的状态(state)
			*/
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

			vkBeginCommandBuffer(commandBuffers[i], &beginInfo);

			// Render pass 通过vkCmdBeginRenderPass 开始后绘画才vkCmdBindVertexBuffers能开始
			//需要VkRenderPassBeginInfo来描述Render Pass 的一些细节
			VkRenderPassBeginInfo renderPassInfo = {};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = renderPass;
			renderPassInfo.framebuffer = swapChainFramebuffers[i];
			//std::cout << "swapChainFramebuffers:" << swapChainFramebuffers.size() << std::endl;

			/*
			限定render 的区域，它定义了着色器(Shader)加载(load)和存储(store)的发生的区域，
			区域外属于未定义部分，为了获得更好的性能，render区域应和attachment的尺寸一致
			*/
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = swapChainExtent;

			//因为现在对于 VK_ATTACHMENT_LOAD_OP_CLEAR 有很多附件，所以需清除附件信息  
			std::array<VkClearValue, 2> clearValues = {};
			clearValues[0].color = { 0.5f, 0.5f, 0.5f, 1.0f };//用黑色清空frame buffer ,对应我们之前设置的VK_ATTACHMENT_LOAD_OP_CLEAR参数
			clearValues[1].depthStencil = { 1.0f, 0 };

			renderPassInfo.clearValueCount = clearValues.size();
			renderPassInfo.pClearValues = clearValues.data();

			//所有的记录命令都以vkCmd..前缀开始，第一个参数都是要将命令记录的位置，即CommandBuffer
			/*
			第三个参数控制带有Render Pass的绘画命令(drawing command)如何被处理，取值:

			VK_SUBPASS_CONTENTS_INLINE: Render pass commands 嵌入到 command buffer中，
			secondary command buffers 将不会被执行。
			VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS: Render pass commands 将从
			secondary command buffers 中被执行。
			*/
			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);//开始绘制的命令

			/*
			VK_PIPELINE_BIND_POINT_GRAPHICS属于绘制类型，VK_PIPELINE_BIND_POINT_COMPUTE 属于计算类型。
			我们的Pipeline是图形类型，所以选VK_PIPELINE_BIND_POINT_GRAPHICS，
			他能控制如下命令:vkCmdDraw, vkCmdDrawIndexed, vkCmdDrawIndirect, and vkCmdDrawIndexedIndirect等。
			*/
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);//将commnand buffer 和 pipeline 绑定:

			VkBuffer vertexBuffers[] = { vertexBuffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

			vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

			vkCmdDrawIndexed(commandBuffers[i], indices.size(), 1, 0, 0, 0);

			vkCmdEndRenderPass(commandBuffers[i]);

			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {//结束记录command buffer
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}
	/*
	在Vulkan中可以使用两种方法进行同步:fences 和 semaphore ,他们都能够使一个操作发送信号(signal)，
	另一个操作等待(wait) fence或者semaphore, 最终使的fence 或semaphore从unginaled 状态变为signaled状态。
	所不同的是fence 可以在程序中使用vkWaitForFences()来获取状态，而semaphore则不可以。Fence 主
	要在渲染操作时同步应用自身(synchronize your application itself with rendering operation)，
	而semaphore被设计为同步一个或跨多个命令队列工作。而我们想要同步绘画命令的队列操作和显示命令的队列操作，所以semaphore更为适合。
	*/
	void createSemaphores() {
		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, imageAvailableSemaphore.replace()) != VK_SUCCESS ||
			vkCreateSemaphore(device, &semaphoreInfo, nullptr, renderFinishedSemaphore.replace()) != VK_SUCCESS) {

			throw std::runtime_error("failed to create semaphores!");
		}
	}

	void updateUniformBuffer() {
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() / 1000.0f;

		UniformBufferObject ubo = {};
		ubo.model = glm::rotate(glm::mat4(), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;

		void* data;
		vkMapMemory(device, uniformStagingBufferMemory, 0, sizeof(ubo), 0, &data);//
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, uniformStagingBufferMemory);

		copyBuffer(uniformStagingBuffer, uniformBuffer, sizeof(ubo));
	}
	/*
	drawFrame()要做如下几件事:

    从Swap Chain 请求一个image。

    执行带有这个image的command buffer ，这个image曾被当做attachment存储在framebuffer中(Execute the command buffer with that image as attachment in the framebuffer)。
    将image 返回到swap chain 等待显示。

	*/
	//虽然所有的操作都在一个函数里运行，但是它们的执行却是异步的

	//该函数用于将数据显示到屏幕上
	void drawFrame() {
		uint32_t imageIndex;
		//std::cout << "imageIndex " << std::endl;

		//获取swap chain里的image,因为swap chain是一个扩展功能，所以需要使用vk***KHR命名约定
		VkResult result = vkAcquireNextImageKHR(device, swapChain,
			//timeout表示等待时间，单位纳秒,如果timeout=0,函数不被阻塞，立刻返回:成功或者VK_NOT_READY。使用64位整数的最大值表示一直等待，直到获得数据
			std::numeric_limits<uint64_t>::max(),
			imageAvailableSemaphore, 
			VK_NULL_HANDLE,
			&imageIndex);// pImageIndex表示可使用的Image索引，
		 // imageIndex   : Get the index of the next available swapchain image:
		

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		VkSubmitInfo submitInfo = {};//通过VkSubmitInfo来配置Command buffer 提交到队列和同步控制:
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
		//waitStages 表示pipeline将在何处等待，我希望当image 可以访问时，将颜色写入image，
		//所以定义stage为pipeline的写color atatchement.
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
		//对应于swapChainImages 数组，我们将用这个imageIndex索引来选择合适的Command buffer 。

		
		//当Command buffer 执行完成后，哪一个semaphore发送信号(signal).
		VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		//最后一个参数为可选的fence,这里置为空，因为我们使用的是semaphore
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}//提交Command buffer 到图形队列:

		//绘画帧(drawing a frame)的最后一步就是把绘画结果返回到Swap Chain里，以致最终显示到屏幕上。显示(Presentation) 通过VkPresentationKHR来配置:
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;//pWaitSemaphores表示显示前需要等待的信号量(semaphore),如同VkSubmitInfo。

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;//pSwapchains表示将要把image 提交到的Swap Chain的数组
		//pImageIndices表示要提交到Swap Chain的image索引数组，总是一个数据(pImageIndices中的每一个元素对应pSwapchains中的每一个元素)。
		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);
		//表示每个Swap Chain的显示(Presentation)结果是否正确，它是一个VkResult数组

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}
	}
	//把着色器程序传递到Pipeline之前，我们需要把它们包裹成:VkShaderModule
	void createShaderModule(const std::vector<char>& code, VDeleter<VkShaderModule>& shaderModule) {
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = (uint32_t*)code.data();

		if (vkCreateShaderModule(device, &createInfo, nullptr, shaderModule.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}
	}

	/*
	VK_FORMAT_B8G8R8A8_UNORM 表示通过8字节无符号整型（总为32字节每个像素）存储bgra通道

	对于surfaceFormat的需求
	format 为:VK_FORMAT_B8G8R8A8_UNORM ，因为这种颜色比较通用;
	colorSpace 为:VK_COLOR_SPACE_SRGB_NONLINEAR_KHR，即支持SRGB颜色。
	*/
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
			return{ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };//按需求返回
		}

		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}//不能自由选择，则查找我们需要的format
		}

		return availableFormats[0];//直接返回第一个
	}

	/*
	VkPresentModeKHR可取的值：

	VK_PRESENT_MODE_IMMEDIATE_KHR = 0, //单  ，立即显示，可能出现撕裂
	VK_PRESENT_MODE_MAILBOX_KHR = 1, // 三缓冲
	VK_PRESENT_MODE_FIFO_KHR = 2,  //双  //可避免撕裂
	VK_PRESENT_MODE_FIFO_RELAXED_KHR = 3, //类似 FIFO

	
	*/
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;//如果不能支持mailbox，则选择fifo
	}

	/*
	extent 是Swap Chain中image的分辨率(resolution) ,通常它与window的尺寸一样。
	vulkan让我们通过设置currentExtent设置width和height来匹配window的分辨率。
	但是有些window Manager会将currentExtent设置为uint32_t的最大值，来表示允许我们设置不同的值，
	这个时候我们可以从minImageExtent和maxImageExtent中选择最匹配window的尺寸值
	*/
	/*
	为swap选择合适的尺寸
	当currentExtent.width/height 任何一个值不是uint32_t的最大值时，直接选择Capabilities.currentExtent, currentExtent最符合window尺寸。

	否则，从Window 尺寸和支持的最大尺寸(maxImageExtent)取最小的，因为尺寸不能超过window，也不能超过capabilities支持的最大尺寸。
	最后为了得到最好的效果，取结果值和minImageExtent的最大值
	*/
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		else {
			VkExtent2D actualExtent = { WIDTH, HEIGHT };

			actualExtent.width = std::max(capabilities.minImageExtent.width,
				std::min(capabilities.maxImageExtent.width, actualExtent.width));
			actualExtent.height = std::max(capabilities.minImageExtent.height, 
				std::min(capabilities.maxImageExtent.height, actualExtent.height));

			return actualExtent;
		}
	}

	//检查显卡是否支持swapchain  获得支持的细节，Swap Chain还可能和我们的window surface不兼容
	/*
	

    Surface 的性能(Capabilities)(比如 : min/max number of images in swap chain, min/max width and height of images)。

    Surface 的格式(formats)(比如 : pixel format, color space)。

    可用的显示模式(present mode)。

	*/
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);//获得VkSurfaceCapabilitiesKHR

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);//获得VkSurfaceFormatKHR 

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());//获得所有支持的VkSurfaceFormatKHR的data
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);//获得VkPresentModeKHR

		//std::cout << "presentModeCount:" << presentModeCount << std::endl;

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());//获得所有支持的VkPresentModeKHR的data
		}

		return details;
	}
	//检查是否是我们需要的
	/*
	显卡选用需具备以下功能：
	1、支持geometry shader
	2、支持图形处理命令
	*/
	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);//获得队列

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		//只要有一个format和presentMode我们就认为此显卡支持Swap Chain 并且兼容surface

		if (extensionsSupported) {//显卡支持的条件下检测swapchain 
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);//查询swapchain支持  当format和presentmode都不为空时，表示交换链支持
			//如果swapChainAdequate的值为true.则可以肯定Swap Chain已经被支持了。而且Swap Chain 的所有特性支持也都被写入swapChainSupport变量中了
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return indices.isComplete() && extensionsSupported && swapChainAdequate;
	}
	//检查支持的扩展
	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;//将可扩展的number存储到extensionCount变量中
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);//存储扩展的具体细节
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
		//ExtensionProperties 提供 the name and version of an extension

		//deviceExtensions为需被支持的swapchain扩展名
		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		//判断swapchain扩展是否被显卡支持
		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}
		//检查显卡是否支持Swap Chain还不够，因为Swap Chain还可能和我们的window surface不兼容，所以还需查询swapchain的支持细节，比创建Instance和Logical Device复杂
		return requiredExtensions.empty();
	}

	//家族队列
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {//需要queueFamily支持VK_QUEUE_GRAPHICS_BIT
				indices.graphicsFamily = i;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);//是否支持显示

			if (queueFamily.queueCount > 0 && presentSupport) {
				indices.presentFamily = i;
			}

			if (indices.isComplete()) {
				break;
			}

			i++;
		}

		return indices;
	}
	//获得extension列表
	std::vector<const char*> getRequiredExtensions() {
		std::vector<const char*> extensions;

		unsigned int glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);//使用glfw来获得

		for (unsigned int i = 0; i < glfwExtensionCount; i++) {
			extensions.push_back(glfwExtensions[i]);
		}

		if (enableValidationLayers) {//添加 VK_EXT_DEBUG_REPORT_EXTENSION_NAME
			extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		}

		return extensions;
	}
	//检查是否支持验证层
	//
	//LunarG validation layers 只有在安装了LunarG SDK的PC上才能使用,这里我们不需要所有的layers，LunarG SDK 只需要VK_LAYER_LUNARG_standard_validation 即可
	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	static std::vector<char> readFile(const std::string& filename) {//获得着色器数据存入到td::vector<char>中
		/*
		ate: Start reading at the end of the file
		binary: Read the file as binary file (avoid text transformations)

		*/
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}
	//VKAPI_ATTR VkBool32 确保有能被vulkan调用的有效签名
	/*
	第一个参数可为以下值
    VK_DEBUG_REPORT_INFORMATION_BIT_EXT
    VK_DEBUG_REPORT_WARNING_BIT_EXT
    VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT
    VK_DEBUG_REPORT_ERROR_BIT_EXT
    VK_DEBUG_REPORT_DEBUG_BIT_EXT

	objType 指定obj的对象的主题消息 eg： if obj is a VkPhysicalDevice then objType would be VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT

	Msg 参数就是我们需要的debug信息，userData是我们要传给debugCallback的数据

	*/
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t obj, size_t location, 
		int32_t code, const char* layerPrefix, const char* msg, void* userData) {
		std::cerr << "validation layer: " << msg << std::endl;

		return VK_FALSE;
	}
};

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	}
	catch (const std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}


/*
loader obj 
加载模型时，需引入tinyobjloader.h
*/
/*
读取纹理时，需引入stb
*/
