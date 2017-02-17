#include "hpc_util.h"
#include "VDeleter.h"

#include <iostream>
#include <stdexcept>//�����쳣
#include <functional>// be used for a lambda functions in the resource management section
#include <chrono>//��¼ʱ��
#include <fstream>//�ļ���
#include <algorithm>//??
#include <vector>
#include <cstring>
#include <array>
#include <set>
#include <unordered_map>  //ȥ���ظ���obj��������


const int WIDTH = 800; //glfw  window  size
const int HEIGHT = 600;

const std::string MODEL_PATH = "models/chalet.obj";//����ģ�͵�·��
const std::string TEXTURE_PATH = "textures/chalet.jpg";//��ȡ�����·��  ����ͼƬ�ĳߴ�Ϊ2^n  һ������������

const std::vector<const char*> validationLayers = {
	"VK_LAYER_LUNARG_standard_validation"
};//validation layers ͨ��ָ�����ǵ�����������

//physics device����֧��������չ���ܴ���swapchain
const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

//VkDebugReportCallbackCreateInfoEXTΪ��չ�ĺ��������� �����Զ����أ�ֻ��ʹ��vkGetInstanceProcAddr���������ַ
VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) {
	auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pCallback);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

// VkDebugReportCallbackEXT ������Ҫ��vkDestroyDebugReportCallbackEXT���
void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
	if (func != nullptr) {
		func(instance, callback, pAllocator);
	}
}

struct QueueFamilyIndices {
	int graphicsFamily = -1;//ͼ�������
	int presentFamily = -1;//��ʾ����

	bool isComplete() {
		return graphicsFamily >= 0 && presentFamily >= 0;
	}
};
/*
swapchain������vulkan�ﱻ��ʾ�Ĵ���
//swapchainΪ����ʾ��ͼƬ����
swapchainͨ��ˢ����Ļ��������ʾͼƬ

Surface ������(Capabilities)(���� : min/max number of images in swap chain, min/max width and height of images)��

Surface �ĸ�ʽ(formats)(���� : pixel format, color space)��

���õ���ʾģʽ(present mode)��


*/
//
struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;//����
	std::vector<VkSurfaceFormatKHR> formats;//��ʽ
	std::vector<VkPresentModeKHR> presentModes;//��ʾģʽ
};

struct Vertex {
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;//���ֱ��ʹ���������굱��rgbsֵ   ��gΪ������  rΪ������

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription = {};
		bindingDescription.binding = 0;//ָ����binding�����������ֵ
		bindingDescription.stride = sizeof(Vertex);
		/*
		inputRate  ������ֵ��ѡ   
		VK_VERTEX_INPUT_RATE_VERTEX �� Move to the next data entry after each vertex��ÿ�������
		VK_VERTEX_INPUT_RATE_INSTANCE  ��Move to the next data entry after each instance��ÿ��ʵ����һ��ֻ�ڼ�����ɫ����ʹ�ã����ڶ�����ɫ��������ʵ�����󣩣�
		*/
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}
	//һ����������һ��VKVertexInputAttributeDescription
	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};
		attributeDescriptions[0].binding = 0;  
		attributeDescriptions[0].location = 0;//����ֵ
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);//���texCoord��Vertex���ƫ����

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

	VDeleter<VkInstance> instance{ vkDestroyInstance };//vkDestoryInstance���� clean up the instance
	VDeleter<VkDebugReportCallbackEXT> callback{ instance, DestroyDebugReportCallbackEXT };//���destory����
	/*
		 vulkan ƽ̨�޹� ������ʹ��WSI��չ������vulkan�ʹ���ϵͳ
	VK_KHR_surface��һ��Instance �������չ�������ڴ���Instanceʱ�Ѿ�ͨ��
	glfwGetRequiredInstanceExtensions�����������չ
	 The window surface needs to be created right after the instance creation, 
	because it can actually influence the physical device selection
	��ʹvulkanʵ��֧��WSI,���ǲ���ζ��ÿ���Կ���֧�֣�ָ���Կ���Ѱ��һ�־��н���Ⱦ����ύ(presenting)��surface�ϵ�����Ķ���(queue family)��
																						   */
	VDeleter<VkSurfaceKHR> surface{ instance, vkDestroySurfaceKHR };

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;//ѡ���Կ���֧��������Ҫ��һЩ����  VkPhysicalDevice ��ͬInstanceһͬ���٣����ﲻ��ʹ��VDeleter
													 /*
													 �Կ�Type:
													 VK_PHYSICAL_DEVICE_TYPE_OTHER = 0, //other
													 VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1, //����
													 VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,  //����
													 VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3, //����
													 VK_PHYSICAL_DEVICE_TYPE_CPU = 4,  //running on cpu
													 */
	VDeleter<VkDevice> device{ vkDestroyDevice };//����ɾ��logical deivce  ֻ��physical device�ǲ��еģ�������logical device��������  logical device������instance����
	VkQueue graphicsQueue;//ͼ������У����ƣ���Logical Device������һ�𴴽���ɾ��
	VkQueue presentQueue;//��ʾ����

	VDeleter<VkSwapchainKHR> swapChain{ device, vkDestroySwapchainKHR };//����swapchain
	std::vector<VkImage> swapChainImages;//swapChain���image��swapChainһ�𴴽���һ������
	VkFormat swapChainImageFormat;//image�ĸ�ʽ
	VkExtent2D swapChainExtent;//image�ķֱ���
	//Ϊ��ʹ��VkImage,��������Swap Chain ������Pipeline �У����Ƕ����봴��VkImageView,
	//����ͬ����������˼һ��,imageView��image��һ�� view.��������������η���image������image����һ���ֵ�
	std::vector<VDeleter<VkImageView>> swapChainImageViews;//����vkimageView,����color attachment
	std::vector<VDeleter<VkFramebuffer>> swapChainFramebuffers;

	VDeleter<VkRenderPass> renderPass{ device, vkDestroyRenderPass };
	VDeleter<VkDescriptorSetLayout> descriptorSetLayout{ device, vkDestroyDescriptorSetLayout };
	VDeleter<VkPipelineLayout> pipelineLayout{ device, vkDestroyPipelineLayout };
	VDeleter<VkPipeline> graphicsPipeline{ device, vkDestroyPipeline };

	VDeleter<VkCommandPool> commandPool{ device, vkDestroyCommandPool };

	//depth buffering ���ڴ洢���ֵ
	VDeleter<VkImage> depthImage{ device, vkDestroyImage };
	VDeleter<VkDeviceMemory> depthImageMemory{ device, vkFreeMemory };
	VDeleter<VkImageView> depthImageView{ device, vkDestroyImageView };

	//create the actual texture image  image and its memory handle
	VDeleter<VkImage> textureImage{ device, vkDestroyImage };
	VDeleter<VkDeviceMemory> textureImageMemory{ device, vkFreeMemory };
	VDeleter<VkImageView> textureImageView{ device, vkDestroyImageView };

	//����sampler����  sampler����ҪVkImage  ��ֱ�Ӵ�texture��ȡcolors   1D/2D/3D texture������ʹ��
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
	// image �Ѿ��õ������Ա���Ⱦ��
	VDeleter<VkSemaphore> imageAvailableSemaphore{ device, vkDestroySemaphore };
	// image ��Ⱦ��Ͽ��Ա��ύ��ʾ��
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
		loadModel();//objģ�ͼ���
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

		vkDeviceWaitIdle(device);// �ȴ�����ĳ�����������ĵ�ĳ�������Ľ���
	}
	
	static void calculate()
	{
		//auto currentTime = std::chrono::high_resolution_clock::now();
		//float time = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() / 1000.0f;


		/*std::cout << "startTime ��" << startTime << std::endl;
		std::cout << "currTime ��" << currTime << std::endl;
		*/
		if (nCount == 0)
		{
			nCount++;
			startTime = glfwGetTime();
		}
		else if ((currTime - startTime) > 1.0)
		{
			std::cout << "֡���ʣ�" << nCount << std::endl;
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
	��֤�����Ĺ���

	�������Ƿ����á�
	׷�ٶ���Ĵ��������٣�����Ƿ�������Դй¶��
	׷���߳�(thread)���õ�Դͷ������߳��Ƿ�ȫ��
	���������õĲ�����ӡ����׼�����
	Tracing Vulkan calls for profiling and replaying��

	������Ѷѷ���Щ��֤��������������Ȥ�ĵ��Թ��ܡ�
	*/
	
	//����vkinstance
	void createInstance() {
		//������֤  �ڷ����Ĵ����п��Բ�����validationLayer
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}
		//����Ӧ�õ���Ϣ
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;//����
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);//����
		appInfo.apiVersion = VK_API_VERSION_1_0;//����

		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = extensions.size();
		createInfo.ppEnabledExtensionNames = extensions.data();

		if (enableValidationLayers) {//�п�֧�ֵ�validation layer
			createInfo.enabledLayerCount = validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}
		//replace ���� ����clean up for any existing handle and then gives you a non-const pointer to overwrite the handle
		//instance����洢��VKInstance�ĳ�Ա��
		if (vkCreateInstance(&createInfo, nullptr, instance.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}
	}
	/*
	��create function��  eg:vkCreateInstance
	Pointer to struct with creation info
	Pointer to custom allocator callbacks, always nullptr in this tutorial
	Pointer to the variable that stores the handle to the new object
	*/

	void setupDebugCallback() {
		if (!enableValidationLayers) return;

		VkDebugReportCallbackCreateInfoEXT createInfo = {};//Ϊcallback����������Ϣ   
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;//����debug  error��warning
		createInfo.pfnCallback = debugCallback;//pUserData Ҳ����ָ�����ݣ�����userData

		if (CreateDebugReportCallbackEXT(instance, &createInfo, nullptr, callback.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug callback!");
		}
	}

	//����surface  ������Ⱦ����ʾ
	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, surface.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}
	//ѡ�������豸���Կ���
	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());//ö�ٳ���ǰ�豸���õ������Կ�

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
	//����Logical Device
	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);//��ʾ��ȡ�ĺ��ʵ��Կ�������ֵ

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;//�豸���д���������Ϣ
		std::set<int> uniqueQueueFamilies = { indices.graphicsFamily, indices.presentFamily };//���ƺ���ʾ����

		float queuePriority = 1.0f;
		for (int queueFamily : uniqueQueueFamilies) {
		std::cout << "queueFamily" << uniqueQueueFamilies.size() << std::endl;

			VkDeviceQueueCreateInfo queueCreateInfo = {};//����queueCount��queueFamilyIndex���͵Ķ���
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; 
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;//Ŀǰֻ��Ҫһ��queue family
			queueCreateInfo.pQueuePriorities = &queuePriority;//���е����ȼ� 0-1
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures = {};  //���Բ���Ĭ��ֵ

		VkDeviceCreateInfo createInfo = {};//�Զ���(queue)������(features)֧�ֵ��޶��⣬���ж�Validation layers �� Extensions���޶���
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos.size();

		createInfo.pEnabledFeatures = &deviceFeatures; 

		//enableValidationLayers��validationLayersֱ��ȡ�Դ���VkInstancesʱ���еĶ���
		createInfo.enabledExtensionCount = deviceExtensions.size();
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		//����logical device
		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, device.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		/*
		device : logical device.
		indices.graphicsFamily : �������ࡣ
		queueIndex : ������ 0 ����Ϊֻ������һ�����У�������������Ϊ0.
		VkQueue * : &graphicsQueue��
		*/
		vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);//��ȡ���� handle
		vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
	}

	//����������
	/*
	����Ϊ�����ṩҪ��Ⱦ��ͼƬ��Ȼ����Ⱦ��Ĺ������Ե���Ļ�ϡ�
	Swap Chain���뱻Vulkan��ʾ�Ĵ������ӱ����Ͻ���Swap Chain����һ��ͼƬ�Ķ���(a queue of images),�����ͼƬ���ű���ʾ����Ļ�ϡ�
	���ǵ�Ӧ�ý�����һ��ͼƬ��Ȼ��滭����֮�����ύ��������ȥ��
	Swap Chain ͨ����������ͨ����Ļˢ����(refresh rate of the screen)��ͬ������ͼƬ����ʾ��

	Ϊ��Ѱ����ѵ�Swap Chain���ã����Ǿ���������������������:

    Surface (��ʽ)format (��:color depth)
    ��ʾģʽ(Presentation mode)(��:��Ⱦ���ͼƬ������������ʾ����ʱ��).
    �����Ĵ�С(Swap extent)(��:ͼƬ��swap chain��ķֱ���)

	*/
	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);//���֧�ֵ�ϸ��

		//Ѱ����ѵ�swapchain����
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);//get format
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);//get presentmode
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);//get extent  image�ķֱ���
	//	std::cout << "imageCount:" << swapChainSupport.capabilities.minImageCount << std::endl;
		//minImageCount ��image �Ѿ�ʮ�ֺ����ˣ�����Ϊ�˸��õ�֧�������壬�����ֶ����һ����maxImageCount ���Ϊ0�� ��ʾ��������������κ����ơ�

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;//����Swap Chain ��image����������������ָ���еĳ���:
		std::cout << "imageCount:" << imageCount << std::endl;

		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {//��maxImageCount =0
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
		createInfo.imageArrayLayers = 1;//image�Ĳ��  һ��Ϊ1 ���Ǵ���3DӦ��
		/*
		imageUsage��
		�����������Ⱦһ��������ͼƬȻ���ٽ��д����Ǿ�Ӧ��ʹ��VK_IMAGE_USAGE_TRANSFER_DST_BIT��ʹ���ڴ�ת����������Ⱦ�õ�image ת����SwapChain�
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT ��ʾimage������������VkImageView����VkFrameBuffer���ʺ�ʹ��color ���� reslove attachment.*/
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;//��ʾ��swapchain��ʲô����  ������Ҫ�Ƕ�image������Ⱦ��image��������ɫ����������

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { (uint32_t)indices.graphicsFamily, (uint32_t)indices.presentFamily };

		if (indices.graphicsFamily != indices.presentFamily) {
			/*
			���grapics queue �� present queue����ͬ���ͻ��������ֶ��з���image�������
			������grapics queue �л滭image,Ȼ�����ύ��presention queue ȥ�ȴ���ʾ
			imageSharingModeȡֵ��
			VK_SHARING_MODE_EXCLUSIVE : image һ��ʱ����ֻ������һ�ֶ��У�����Ȩ��ת��������ȷ���������ѡ������ṩ�Ϻõ����ܡ�
			VK_SHARING_MODE_CONCURRENT : image ���Կ���ֶ���ʹ�ã�����Ȩ��ת��������ȷ������
			*/
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;//imageSharingMode ��ʾ���ֶ����У�image���ʹ��
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;//����Image  �任
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;// ���Ժ�����������ɫ���ʱ��Alpha ͨ��
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;// ��������Щ���ڸǵ����� 

		VkSwapchainKHR oldSwapChain = swapChain;
		createInfo.oldSwapchain = oldSwapChain;

		VkSwapchainKHR newSwapChain;
		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &newSwapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}
		swapChain = newSwapChain;
		//���swapchain�������image
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		//std::cout << "imageCount:" <<imageCount << std::endl;

		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;//����format��ʽ
		swapChainExtent = extent;//����extent
	}

	//Ϊswap chain���image����imageview
	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size(), VDeleter<VkImageView>{device, vkDestroyImageView});

		for (uint32_t i = 0; i < swapChainImages.size(); i++) {//Ϊÿ��image����imageview
			createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, swapChainImageViews[i]);
		}
	}

	/*
	����Render Pass ��ҪSubpass ��Attachment, ���ھͼ򵥵����ΪRender Pass ����Subpass ����� 
	Attachment����ɡ�Render Pass�Ĺ�����ҪSubpass ����ɣ� ÿһ��Subpass ���Բ������Attachment ,
	��ô��Attachment�����б�ʾ��Щattachment�ᱻĳ��Subpass�����أ�
	����������Ҫһ��VkAttachmentReference������attachment��Attachment�����е��±�ʹ���ʱ��layout��
	*/

	/*
	�ڴ���Pipeline ֮ǰ���Ǳ������Vulkan����ȾʱҪʹ�õ�FrameBuffer ����(attachments)����Ҫ����ʹ��color buffer �Լ�
	depth buffer attachments��������
	Ҫʹ�ö��ٸ�����(samples)�Լ�Ӧ����δ�����������ݡ�������Щ��Ϣ��������д��Render Pass��
	*/
	void createRenderPass() {
		VkAttachmentDescription colorAttachment = {};
		colorAttachment.format = swapChainImageFormat;//��ɫ������ʽ������swapchain���һ��imageƥ��
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;//��ʹ�ÿ���ݣ�����Ϊ1

		// subpass ִ��ǰ��color��depth attachment���������ִ���
		/*loadOp��storeO��ʾ��Ⱦǰ����Ⱦ��Ҫ���Ķ����������ǵ�������,д����Ƭԭ(fragment)֮ǰ�����FrameBuffer,ʹFrameBuffer��Ϊ��ɫ��
		����������Ⱦ�����������ʾ����Ļ�ϣ������������ǽ�storeOp ����Ϊ���档*/
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		/*
		
		VK_ATTACHMENT_STORE_OP_STORE: Rendered contents will be stored in memory and can be read later
		VK_ATTACHMENT_STORE_OP_DONT_CARE: Contents of the framebuffer will be undefined after the rendering operation

		*/
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		/*
		��Vulkan�У��þ����ض����ظ�ʽ��VkImage ��ʾ����(texture)��FrameBuffer,���������ڴ��еĲ���(layout)��������ʹ��Image 
		��Ŀ�Ĳ�ͬ�ǿ��Ըı�ġ�һЩ������layout��:

		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: Images ���� color attachment
		VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: ��ʾһ��Ҫ����ʾ��swap chain imageԴ
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: images ��Ϊ�ڴ濽��������Ŀ��
		*/
		/*
		initial / finalLaout��ʾ��Ⱦǰ��image��layout, VK_IMAGE_LAYOUT_UNDEFINED��ʾ���ǲ��ں�֮ǰ��layout,
		��VK_IMAGE_LAYOUT_PRESENT_SRC_KHR��ʾ����Ⱦ��ʹ��Swpa Chain ʱ��image ���ڿ���ʾ��layout
		*/
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentDescription depthAttachment = {};//depth attachment
		depthAttachment.format = findDepthFormat();//�ҵ����ʵ�format  ������ depth image ��ͬ
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;//don't care now   ����Ӳ��ִ�ж���Ĳ���
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		//�����render������ The layout of the imageû�б仯����initial��final��ͬ
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		//ÿһ��Subpass ����һ������������ǰһ����VkAttachmentDescription ������attachment(s) ,ÿһ��������VkAttachmentReference����:
		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;//ֵ��һ������(index)  ��ʾ�����ĸ�����
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef = {};//��Ϊsubpass��ӵ���ȸ���
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		/*
		һ��Render Pass ��һϵ�е�subPass��ɣ�����subpass ����������Ⱦ����,������Ⱦ��һ���׶�,
		��Ⱦ����洢��һ��Render Pass�����subpass�У�һ��subpass�Ķ���ȡ������һ��subpass �Ĵ�������
		������ǰ����Ǵ����һ��Render Pass ,Vulkan �ܹ�Ϊ���������������ǵ�ִ��˳�򣬽�ʡ�ڴ�����Ӷ����ܻ�ȡ���õ�����
		*/
		//pipelineBindPoint��ʾҪ�󶨵�Pipeline���ͣ�����ֻ֧����������:����(compute)��ͼ��(graphics)����������ѡ��ͼ��Pipeline
		VkSubpassDescription subPass = {};
		subPass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subPass.colorAttachmentCount = 1;
		subPass.pColorAttachments = &colorAttachmentRef;
		subPass.pDepthStencilAttachment = &depthAttachmentRef;
		/*
		
		Subpass ���������õ�attachment��:

		pInputAttachments: ����ɫ���ж�ȡ�� attachment
		pResolveAttachments: multiple color attachment
		pDepthStencilAttachment: depth and stencil data ��attachment
		pPreserveAttachments: ����Subpass ʹ�ã�������ĳ��ԭ����Ҫ����

		*/


		/*
		ender Pass �е� subpass �Զ�����image (attachment)��layoutת����
		��Щת����subpass ����(subpass dependencies) ������,��ָ����subpass����ڴ��ִ��������
		��Ȼ����ֻ��һ��subpass,������ִ�����subpass��ǰ�����Ҳ����ʽ�ĵ���subpass�ˡ�

		���������õ�����(built-in dependencies)����render passǰ��render pass���ת����
		ǰ�߳��ֵ�ʱ��������ȷ����Ϊ���ٶ�render passǰ��ת��������pipeline��ʼ��ʱ��
		�������ʱ�����ǻ�û�л��image�أ���Ϊimage����VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT�׶�(stage)�Ż�õġ�
		�������Ǳ�����д/�������������
		*/
		VkSubpassDependency dependency = {};
		/*
		srcSubpass��dstSubpass�ֱ��ʾ�����������ʹ�����subpass(���������������ߵ�����).
		VK_SUBPASS_EXTERNAL����Render pass ǰ����������subpass,
		��ȡ����VK_SUBPASS_EXTERNA�Ǳ�������src����dst��
		����0ָ�����Ƕ���ĵ�һ��ͬʱҲ��Ψһ��һ��subpass��
		dst�������src �Է�ֹѭ��������
		*/
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		/*
		�����������ֶηֱ��������ǽ���ʲô�����ϵȴ��Լ���������ں��ֽ׶η�����
		���Ǳ���ȴ�Swap Chain��image����֮����ܷ��������������������pipeline�����׶Ρ�
		*/
		dependency.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependency.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		/*
		��������������ʾ����д color attachment ���������� _COLOR_ATTACHMENT_ �׶ν��еȴ���
		��Щ���ñ�֤:���Ǳ�Ҫ(�磺�����������д��ɫ(color)��ʱ��)����ת�������ᷢ����
		*/
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		/*
		Attachment �� Subpass ���Ѿ��������ˣ����ڿ�ʼ����Render Pass��
		*/

		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };//������ɫ�������������
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
		
		����descriptor��Ҫ   descriptor layout, descriptor pool and descriptor set 
		*/
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		//create a new descriptor "combined image sampler" ��֤shader���Խ���imageͨ��sampler
		VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;//stageFlags��ʾ combined image sampler descriptor ������shaderʹ��

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
		auto vertShaderCode = readFile("shaders/vert.spv");//�����ɫ���Ķ������ļ�
		auto fragShaderCode = readFile("shaders/frag.spv");

		VDeleter<VkShaderModule> vertShaderModule{ device, vkDestroyShaderModule };
		VDeleter<VkShaderModule> fragShaderModule{ device, vkDestroyShaderModule };
		createShaderModule(vertShaderCode, vertShaderModule);
		createShaderModule(fragShaderCode, fragShaderModule);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		//ָ��ʹ�ö�����ɫ��������main������ʼ����
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

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};//VkPipelineVertexInputStateCreateInfo ��ʾ���ݶ���ĸ�ʽ
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		/*

		�������ݵ�����(Bindings) :���ݼ�ļ����
		�Լ��ж������Ƕ�������(pre-vertex)����ʵ������(pre-instance)��

		�������Ե�������Attribute Descriptions)�����뵽Vertex Shader 
		�������(attributes)���ͣ����ĸ�Binding�����Լ�offset��
		
		*/
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		//��ʲô���ļ���ͼ�κ�ͼԪ�����Ƿ��������
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		/*
		topology  ���ͣ�
		VK_PRIMITIVE_TOPOLOGY_POINT_LIST: ����
		VK_PRIMITIVE_TOPOLOGY_LINE_LIST: ÿ������Ϊһ���ߣ����㲻������
		VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: һ���ߵĵڶ������������Ϊ��һ���ߵ����(������)
		VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: ÿ������һ�������Σ����㲻������
		VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: һ�������εĵ������������Ϊ��һ�������ε����(������)

		*/
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;//ͼԪ�Ƿ�����  ��_STRIP topology modes �й�

		//Viewport ��ʵ��������������Ⱦ��FrameBuffer�Ķ�������С������Ǵ�����(0,0)�㿪ʼ,
		//����һ����(width)�͸�(height)�ľ��������������������VkViewport��ʾ
		/*
		�����ڴ���swpaChainʱ��������������swapChain����Image�ĳߴ���ܺ�window�ĳߴ粻ͬ��������Swap Chain�� width��height ��ֵViewport, 
		��Ϊ������Swap Chain�� images ����ΪFrameBufferʹ�á�Min/maxDepth��ʾFrameBuffer����ȷ�Χ�����ȡֵ��[0.0 , 1.0]��Χ�ڣ�
		ע�⣬minDepth���ܴ���maxDepth,���û��ʲô������Ҫ�����ǽ����ձ�׼�Ķ��壬 ��:minDepth=0 , maxDepth=1.0
		*/
		//Viewport ������image �� FrameBuffer�ı任����Scissor ���ο������Щ��������ؽ��ᱻ�洢��
		//��Scissor���ο�������ؽ����ڹ�դ��(���ػ�)�׶α����������ԣ�����任��Scissor ������һ����������
		VkViewport viewport = {};//�����ӿ�
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};//�������ô���
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		/*
		ע��,��VkPipelineViewportStateCreateInfo�Ľṹ����������ĳЩ�Կ��ϣ����ǿ���ʹ�ö��Viewport�Ͷ��Scissor ��
		���漰���Կ���֧�֣������Ǵ���Logicsl Deviceʱ��
		VkPhysicalDeviceFeatures �ֶ����� VkBool32 multiViewport;�Ķ��壬����Լ���Լ����Կ��Ƿ�֧��������ԡ�
		*/
		VkPipelineViewportStateCreateInfo viewportState = {};//���ӿ�����ô��ڽ������
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		/*
		��դ��(���ػ�)����������Vertex Shader �����󶥵���ɵļ���ͼ����ɢ����һ����Ƭԭ(fragment)��
		Ȼ��Ƭԭ���ݵ�Fragment Shader �������ɫ��
		��դ��Ҳִ��depth testing��face culling �� scissor test����������ã�
		ѡ���ǽ������������ɢ����Ƭԭ������ֻ��ɢ���߿�(edges)
		(�ֽ� : wireframe rending),����ͨ�����½ṹ����������
		*/
		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		/*
		//������ó�VK_TRUE,��Щ���Ӿ����ƽ��(near)��Զƽ��(far)֮���Ƭԭ���ᱻ����/��ȡ
		����Ϊfalse���ʾΪ����
		*/
		rasterizer.depthClampEnable = VK_FALSE;
		/*
		rasterizerDiscardEnable ���ΪVK_TRUE, 
		��������(geometry)���޷�ͨ��Rasterization�׶Σ�
		FrameBuffer ���ò����κ��������
		*/
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		/*
		polygonModeȡֵ��

		VK_POLYGON_MODE_FILL: �����������������Ƭԭ
		VK_POLYGON_MODE_LINE: ֻ�ж���α߽�(edges)��Ƭԭ
		VK_POLYGON_MODE_POINT: ֻ������ζ���
		*/
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;//������ϸ
		/*  ���Ʒ�ʽ�뱳��ü�
			�涨�ü��Ǹ���:ǰ��ͱ��棬��������ĽǶȿ������㰴��ʱ����ɵ�ͼ��
			�����棬˳ʱ���档������ľ��Ʒ�ʽ�����Զ��塣
		*/
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;//��ʾ�ü���ʽ 
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling = {};//���ز�������ִ�з���ݵĹ���
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		//��Ȳ��Թ�����ͨ��VkPipelineDepthStencilStateCreateInfo����
		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		//��������������ʾ��ƬԪ�豻�ȽϺ�д��
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;//�ò����ڻ���͸������ʱ�Ƚ�����
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		//������ȷ�Χ�Ĳ��ԣ���
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		/*
		���(mix) ����ɫ�;���ɫ���������յ���ɫ��

		������ɫ�;���ɫͨ����λ����(bitwise)������һ��

		*/
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | 
			VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		//false  Ϊalphaͨ�����
		colorBlending.logicOpEnable = VK_FALSE;//��λ���(bitewise combination)�� logicOpEnable��Ҫ���ó�VK_TRUE  
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
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;//ֻ��һ��pipeline

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
	�ڴ���Render Passʱ,��������ӵ��һ����Swap Chain ��image������ͬ��ʽ(format)��FrameBuffer

	��attachments������FrameBuffer�У�FrameBuffer ͨ������VkImageView���������е�attachments

	�ڱ�������ֻ��һ��attachment : color attachment ��Ȼ����Ϊattachment��imageȡ��������ʾ��ʱ��Swap Chain
	���׷��ص�����һ��image�������ζ��������ҪΪSwap Chain���ÿһ��image ����һ��FrameBuffer
	*/
	void createFramebuffers() {

		//std::cout << "swapChainImageViews.size() :" << swapChainImageViews.size() << std::endl;
		swapChainFramebuffers.resize(swapChainImageViews.size(), VDeleter<VkFramebuffer>{device, vkDestroyFramebuffer});

		// ����ÿһ��imageView (image) Ϊ������Framebuffer
		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			std::array<VkImageView, 2> attachments = {//����������
				swapChainImageViews[i],
				depthImageView
			};

			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			// ����� attachmentCount ��pAttachments ��Render Pass ���
			// pAttachment  �����
			framebufferInfo.attachmentCount = attachments.size();
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;//swap chain imageֻ��һ��

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, swapChainFramebuffers[i].replace()) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}
	/*
	��Vulkan�У���滭����ڴ�ת���Ȳ���������ֱ��ͨ����������ȥ��ɵģ�������Ҫ�����е�
	��������Command Buffer �
	������һ���ô����ǣ���Щ�����úõľ����ѶȵĻ�ͼ�����������ڶ��̵߳��������ǰ��ɡ�
	*/
	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		//Command pools ����Command buffer ���ڴ����Command buffer ��Command pool�б�����
		VkCommandPoolCreateInfo poolInfo = {};//���ȴ���commandpool
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		/*
		command buffer ��Ҫ�ύ�������еȴ�ִ�У��������Ѿ��õ���ͼ�ζ���(graphics)��
		��ʾ����(presentation)��
		��һ��Command pool ��������command buffers ֻ�ܶ�Ӧһ���ض��Ķ��С�
		��Ϊ������ʹ�û�ͼ�������ѡ��graphics ���� ��
		*/
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;//����ͼ�����

		/*

		flags��ȡֵֻ������:

		VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: Command buffer �Ƚ϶��������ܻ���һ����Խ϶�
		��ʱ���ڱ����û��ͷţ���Ҫ���ڿ���pool���ڴ�ķ�����Ϊ��
		VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: commad buffer�Ƿ���Ա��ֱ����ã�
		����޴˲�����pool�����е�command buffer���������á�

		ֻ���ڳ���Ŀ�ʼʱ��¼Command buffer ,Ȼ����main loop �е��ö�Σ����Բ���Ҫflags����
		*/

		if (vkCreateCommandPool(device, &poolInfo, nullptr, commandPool.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics command pool!");
		}
	}
	//�贴��Image ImageView �Լ�transitionImageLayout
	void createDepthResources() {
		VkFormat depthFormat = findDepthFormat();

		createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, 
			VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
		createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 
			depthImageView);

		//ֻ��ת����layout���ʺ�depth attachment ʹ��
		transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, 
			VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	}

	/*
	 VkFormatProperties��ȡ��������ֵ

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
	//�鿴depth format �Ƿ���stencil component
	bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	//����image ��output texture
	void createTextureImage() {
		int texWidth, texHeight, texChannels;//����Ŀ�߶�  ��  ͨ��
		/*
		load image 
		STBI_rgb_alpha  Ϊstb_image.h�ж����  һ���ڼ�������ʱ��Ҫ����һ��͸���ȣ�����������������
			Ϊ��֤�ͽ���������������һ�£�jpgΪ3ͨ��
		���  ����Ŀ��  ��ͨ������
		stbi_uc  Ϊ�޷���char ������
		pixelsָ�뷵�ص���char array�����Ԫ�ص�ֵ
		*/

		stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = texWidth * texHeight * 4;//4 bytes per pixel

	//	std::cout << "texture size " << sizeof(char) << std::endl;

		if (!pixels) {//�������ʧ�ܣ�
			throw std::runtime_error("failed to load texture image!");
		}

		VDeleter<VkImage> stagingImage{ device, vkDestroyImage };
		VDeleter<VkDeviceMemory> stagingImageMemory{ device, vkFreeMemory };
		//Ϊ���� LINEAR ƽ��չ��   ������õĸ�ʽR8G8B8A8
		createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingImage, stagingImageMemory);
		/*
		the VK_FORMAT_R8G8B8A8_UNORM format is not supported by the graphics hardware
		*/
		//���ڴ��豻�����ɼ���host visible ��ʹ��vkMapMemory
		void* data;
		vkMapMemory(device, stagingImageMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, (size_t)imageSize);
		vkUnmapMemory(device, stagingImageMemory);
		//��image data ���ƽ�memory����clear pixel array
		stbi_image_free(pixels);

		/*final image ����ߴ���stagingImageһ����
		formatҲӦ�ü���stagingImage����Ϊ����ֻ��copy ԭʼ image data �����
		tiling mode ����Ҫһ��
		�����������Ϊת����Ŀ��ͼ�� ������ϣ������ɫ���п��Զ����������ݽ��в���
		Ϊ��ø��õ����ܣ�memoryӦΪdevice local
		*/
		createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | 
			VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

		/*
		��stage image ���Ƹ� textureImage 

		Transition the staging image to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
		Transition the texture image to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
		Execute the image copy operation

		*/
		//VK_IMAGE_LAYOUT_PREINITIALIZED and VK_IMAGE_LAYOUT_UNDEFINED  ���������� oldLayout ��transition image
		transitionImageLayout(stagingImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyImage(stagingImage, textureImage, texWidth, texHeight);
		//��֤������shader���textureImage���в���
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}

	void createTextureImageView() {
		createImageView(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, textureImageView);
	}
	/*
	
	address mode ����
	
    VK_SAMPLER_ADDRESS_MODE_REPEAT: Repeat the texture when going beyond the image dimensions.
    VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: Like repeat, but inverts the coordinates to mirror the image when going beyond the dimensions.
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: Take the color of the edge closest to the coordinate beyond the image dimensions.
    VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: Like clamp to edge, but instead uses the edge opposite to the closest edge.
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: Return a solid color when sampling beyond the dimensions of the image.

	
	*/
	void createTextureSampler() {
		//sampler ͨ��VkSamplerCreateInfo������
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		//mag ��min����  mag  ���� oversampling   min ���� undersampling
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;//�������Թ���  һ�㶼����  �������ܺܲ�
		samplerInfo.maxAnisotropy = 16;//�����ڼ���������ɫ�Ĳ���ֵ����   ֵԽС����Խ�õ���������   Ŀǰû�г���16��Ӳ��֧��
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;//clamp to border ����������image sizeʱ����� black, white or transparent
		/*;//ָ��texture������ϵ  
		Ϊtrue ʱ  within the [0, texWidth) and [0, texHeight) range

		false   [0, 1) range on all axes
		*/
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;//�������  ������һ��ֵ���бȽϣ����������filtering    һ������shadow map
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;//����mipmap

		if (vkCreateSampler(device, &samplerInfo, nullptr, textureSampler.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	void createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, VDeleter<VkImageView>& imageView) {
		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;//viewType ��VK_IMAGE_VIEW_TYPE_1/2/3D
		viewInfo.format = format;
		//Component �ֶ� ����Ĭ��ֵ

		//subresourceRange����image��ʹ��Ŀ�ĺ�Ҫ�����ʵĲ���
		viewInfo.subresourceRange.aspectMask = aspectFlags;//aspectFlags  ȡֵΪ VK_IMAGE_ASPECT_COLOR_BIT
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
		//ͼ��Ĳ���
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		/*һά��ͼ����������洢���ݻ򽥱�����
		��ά��һ������ͼ��
		��ά�Ŀ������ڴ洢�������ֵ
		*/
		imageInfo.imageType = VK_IMAGE_TYPE_2D;//create 1D, 2D and 3D images
		imageInfo.extent.width = width; //extern �ֶ�����ָ��ͼ��ĳߴ磨�����ᣩ
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		//��ǰ�������Ƕ�ά���飬Ҳ��ʹ��mipmapping  ��Ϊ1  
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		
		imageInfo.format = format;//��ʽ
		 //�����Ҫ��image��memory��ֱ�ӻ�ȡ����Ԫ�أ������Linear  tiling
		imageInfo.tiling = tiling;//չ����ʽ  
		/*��ʼ��layout������ѡ��
		һ����undefined  �ڵ�һ��transition(�任)ʱ���ͻ�discard texels
		һ����PREINITIALIZED  �ڵ�һ�α任ʱ������discard texels
		undefiend �ʺ�image��Ϊ��ɫ����Ȼ���ĸ���ʱʹ��  ��Ϊ��renderpass֮ǰ�Ὣ����clear
		�����Ҫ��texture������ݣ�����ʹ��preinitialized
		*/
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
		//usage:��ΪstagingImage���Ḵ�Ƹ�finalImage,��������Ϊtransfer_src
		imageInfo.usage = usage;

		//samples����ز����й�  ֻ��Ҫͼ����Ϊ����������ֻҪһ��sample
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.flags = 0; //flags��sparse image �йأ���ʹ��3dtextureʱ���ɲ���Sparse images�������˷��ڴ�
		//��imageֻ��һ��֧��transfer������queue family ʹ�ã�
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(device, &imageInfo, nullptr, image.replace()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		//�����ڴ��image  
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
	//����layout translate
	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();
		//using an image memory barrier �����layout transition
		//pipeline barrier  ͨ�������ڶ���Դ��ͬ������   ��֤����һ��buffer��ȡ����ʱд����������
		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		//�����care image���ڵ����ݣ�oldLayout �����ʹ��undefined
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		//һ�����queue family ��index�����û�и�������ignored����
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//ָ�������õ�image   �Լ� �������ⲿ�ֵ�image
		barrier.image = image;
		
		//��ʹ��ʹ��stencil attachment ��ҲҪ������뵽depth_image�� layout transitions
		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

			if (hasStencilComponent(format)) {
				barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			}
		}
		else {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
		//image�Ȳ���һ������Ҳ����mipmapping ����level��layer��Ϊ1
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		//barrier��Ҫ������ͬ���ġ�
		/*
		����ָ����Щ�漰����Դ�Ĳ���������barrier֮ǰ����Щ����wait on barrier
		srcAccessMask  dstAccessMask  ֵȡ����old and new layout,
		*/
		/*
		����ת����Ҫ����  �Ժ���Ҫ����ת����ֱ����������뼴��

		Preinitialized �� transfer source: transfer reads should wait on host writes
		Preinitialized �� transfer destination: transfer writes should wait on host writes
		Transfer destination �� shader reading: shader reads should wait on transfer writes
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
			//ֻ�е���һ���ſ���˵�� image ������Ϊdepth�ĸ���ʹ����
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		//���е�pipeline barrier ��������ͬ�ķ������ύ��
		/*
		@commandBuffer //��һ������ָ����pipeline�׶Σ���barrier֮ǰ��Ӧ�÷����Ĳ���
						  
		@(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, )
				��������ָ����pipeline�׶����ֲ������� wait on barrier,��������ϣ���ò�������������
			������pipeline��topλ�þͷ�����src ��dst��ͬ
		@0		//������������Ϊ0/VK_DEPENDENCY_BY_REGION_BIT.  �ڶ���ֵ��ʾ��barrierת����per-region ״̬
			eg:����ζ��ʵ���Ѿ���������Կ�ʼ��ĿǰΪֹд��Ĳ��ֵ���Դ���ж�ȡ
		������������ʾpipeline barrier ���ֿ��õ����� (memory barriers��buffer memory barriers ��
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

		//ָ����һ���ֵ�image��Ҫ�����Ƹ������image����һ����
		VkImageCopy region = {};
		region.srcSubresource = subResource;
		region.dstSubresource = subResource;
		region.srcOffset = { 0, 0, 0 };
		region.dstOffset = { 0, 0, 0 };
		region.extent.width = width;
		region.extent.height = height;
		region.extent.depth = 1;
		//image copy����ͨ��ʹ��vkCmdCopyImage�������
		/*
		ǰ�������ָ��src/dst ��image/layout
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
	ʹ��descriptor�������������棺	

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
		descriptorWrites[0].dstBinding = 0;//�󶨵�����ֵ  Ϊ0
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &bufferInfo;//ʹ��bufferInfo

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = descriptorSet;
		descriptorWrites[1].dstBinding = 1;//�󶨵�����ֵ  Ϊ1
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pImageInfo = &imageInfo;//ʹ��imageInfo
		//����Ϊֹ��������shader��ʹ��  sampler  

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
		��Ϊ�滭�����漰�󶨵���ȷ��VkFrameBuffer,
		��������ҪΪSwap Chain���ÿһ��image ����һ��Command buffer��
		*/
		commandBuffers.resize(swapChainFramebuffers.size());

		//std::cout << "commandBuffers:" << commandBuffers.size() << std::endl;

		VkCommandBufferAllocateInfo allocInfo = {};//����command buffer
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		/*
		level �ֶ��޶���command buffer ����Ҫ�Ļ��Ǵ�Ҫ�ģ���������ȡֵ:
		VK_COMMAND_BUFFER_LEVEL_PRIMARY: �����ύ��������ִ�У������ܴ�����command buffer �е��á�
		VK_COMMAND_BUFFER_LEVEL_SECONDARY: ����ֱ���ύ�����У������Դ���command buffer �е��á� 
		
		����CommandBuffer ����������֮ǰ�Ķ������������������ͬ��ʹ��:vkFreeCommandBuffers(),
		������һ��Command Pool��һ��CommandBuffer����
		*/
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}

		for (size_t i = 0; i < commandBuffers.size(); i++) {

			//ͨ��vkBeginCommandBuffer����ʼ��¼ Command buffer, ����VkCommandBufferBeginInfo������command buffer�ľ����÷� :
			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			/*
			flags �������Ǹ����ʹ��command buffer ,����ȡֵ:

			VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: ֻ�ܱ��ύһ�Σ�֮����ܱ����á�
			VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: ������command buffer ������reder pass�У�
			��command buffer �����Դ�ֵ��
			VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: command buffer �ڵȴ�ִ��ʱ���Ա��ظ��ύ

			����������ʹ��_SIMUTANEOUS_USE_BIT����Ϊ���п�������һ��֡(frame)��δ���꣬
			��һ��֡�Ļ滭������Ѿ��ύ�ˡ�
			pInheritanceInfo��ʾ��command buffer ����command buffer �̳й�����״̬(state)
			*/
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

			vkBeginCommandBuffer(commandBuffers[i], &beginInfo);

			// Render pass ͨ��vkCmdBeginRenderPass ��ʼ��滭��vkCmdBindVertexBuffers�ܿ�ʼ
			//��ҪVkRenderPassBeginInfo������Render Pass ��һЩϸ��
			VkRenderPassBeginInfo renderPassInfo = {};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = renderPass;
			renderPassInfo.framebuffer = swapChainFramebuffers[i];
			//std::cout << "swapChainFramebuffers:" << swapChainFramebuffers.size() << std::endl;

			/*
			�޶�render ����������������ɫ��(Shader)����(load)�ʹ洢(store)�ķ���������
			����������δ���岿�֣�Ϊ�˻�ø��õ����ܣ�render����Ӧ��attachment�ĳߴ�һ��
			*/
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = swapChainExtent;

			//��Ϊ���ڶ��� VK_ATTACHMENT_LOAD_OP_CLEAR �кܶ฽�������������������Ϣ  
			std::array<VkClearValue, 2> clearValues = {};
			clearValues[0].color = { 0.5f, 0.5f, 0.5f, 1.0f };//�ú�ɫ���frame buffer ,��Ӧ����֮ǰ���õ�VK_ATTACHMENT_LOAD_OP_CLEAR����
			clearValues[1].depthStencil = { 1.0f, 0 };

			renderPassInfo.clearValueCount = clearValues.size();
			renderPassInfo.pClearValues = clearValues.data();

			//���еļ�¼�����vkCmd..ǰ׺��ʼ����һ����������Ҫ�������¼��λ�ã���CommandBuffer
			/*
			�������������ƴ���Render Pass�Ļ滭����(drawing command)��α�����ȡֵ:

			VK_SUBPASS_CONTENTS_INLINE: Render pass commands Ƕ�뵽 command buffer�У�
			secondary command buffers �����ᱻִ�С�
			VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS: Render pass commands ����
			secondary command buffers �б�ִ�С�
			*/
			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);//��ʼ���Ƶ�����

			/*
			VK_PIPELINE_BIND_POINT_GRAPHICS���ڻ������ͣ�VK_PIPELINE_BIND_POINT_COMPUTE ���ڼ������͡�
			���ǵ�Pipeline��ͼ�����ͣ�����ѡVK_PIPELINE_BIND_POINT_GRAPHICS��
			���ܿ�����������:vkCmdDraw, vkCmdDrawIndexed, vkCmdDrawIndirect, and vkCmdDrawIndexedIndirect�ȡ�
			*/
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);//��commnand buffer �� pipeline ��:

			VkBuffer vertexBuffers[] = { vertexBuffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

			vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

			vkCmdDrawIndexed(commandBuffers[i], indices.size(), 1, 0, 0, 0);

			vkCmdEndRenderPass(commandBuffers[i]);

			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {//������¼command buffer
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}
	/*
	��Vulkan�п���ʹ�����ַ�������ͬ��:fences �� semaphore ,���Ƕ��ܹ�ʹһ�����������ź�(signal)��
	��һ�������ȴ�(wait) fence����semaphore, ����ʹ��fence ��semaphore��unginaled ״̬��Ϊsignaled״̬��
	����ͬ����fence �����ڳ�����ʹ��vkWaitForFences()����ȡ״̬����semaphore�򲻿��ԡ�Fence ��
	Ҫ����Ⱦ����ʱͬ��Ӧ������(synchronize your application itself with rendering operation)��
	��semaphore�����Ϊͬ��һ�������������й�������������Ҫͬ���滭����Ķ��в�������ʾ����Ķ��в���������semaphore��Ϊ�ʺϡ�
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
	drawFrame()Ҫ�����¼�����:

    ��Swap Chain ����һ��image��

    ִ�д������image��command buffer �����image��������attachment�洢��framebuffer��(Execute the command buffer with that image as attachment in the framebuffer)��
    ��image ���ص�swap chain �ȴ���ʾ��

	*/
	//��Ȼ���еĲ�������һ�����������У��������ǵ�ִ��ȴ���첽��

	//�ú������ڽ�������ʾ����Ļ��
	void drawFrame() {
		uint32_t imageIndex;
		//std::cout << "imageIndex " << std::endl;

		//��ȡswap chain���image,��Ϊswap chain��һ����չ���ܣ�������Ҫʹ��vk***KHR����Լ��
		VkResult result = vkAcquireNextImageKHR(device, swapChain,
			//timeout��ʾ�ȴ�ʱ�䣬��λ����,���timeout=0,�����������������̷���:�ɹ�����VK_NOT_READY��ʹ��64λ���������ֵ��ʾһֱ�ȴ���ֱ���������
			std::numeric_limits<uint64_t>::max(),
			imageAvailableSemaphore, 
			VK_NULL_HANDLE,
			&imageIndex);// pImageIndex��ʾ��ʹ�õ�Image������
		 // imageIndex   : Get the index of the next available swapchain image:
		

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		VkSubmitInfo submitInfo = {};//ͨ��VkSubmitInfo������Command buffer �ύ�����к�ͬ������:
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
		//waitStages ��ʾpipeline���ںδ��ȴ�����ϣ����image ���Է���ʱ������ɫд��image��
		//���Զ���stageΪpipeline��дcolor atatchement.
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
		//��Ӧ��swapChainImages ���飬���ǽ������imageIndex������ѡ����ʵ�Command buffer ��

		
		//��Command buffer ִ����ɺ���һ��semaphore�����ź�(signal).
		VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		//���һ������Ϊ��ѡ��fence,������Ϊ�գ���Ϊ����ʹ�õ���semaphore
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}//�ύCommand buffer ��ͼ�ζ���:

		//�滭֡(drawing a frame)�����һ�����ǰѻ滭������ص�Swap Chain�����������ʾ����Ļ�ϡ���ʾ(Presentation) ͨ��VkPresentationKHR������:
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;//pWaitSemaphores��ʾ��ʾǰ��Ҫ�ȴ����ź���(semaphore),��ͬVkSubmitInfo��

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;//pSwapchains��ʾ��Ҫ��image �ύ����Swap Chain������
		//pImageIndices��ʾҪ�ύ��Swap Chain��image�������飬����һ������(pImageIndices�е�ÿһ��Ԫ�ض�ӦpSwapchains�е�ÿһ��Ԫ��)��
		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);
		//��ʾÿ��Swap Chain����ʾ(Presentation)����Ƿ���ȷ������һ��VkResult����

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}
	}
	//����ɫ�����򴫵ݵ�Pipeline֮ǰ��������Ҫ�����ǰ�����:VkShaderModule
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
	VK_FORMAT_B8G8R8A8_UNORM ��ʾͨ��8�ֽ��޷������ͣ���Ϊ32�ֽ�ÿ�����أ��洢bgraͨ��

	����surfaceFormat������
	format Ϊ:VK_FORMAT_B8G8R8A8_UNORM ����Ϊ������ɫ�Ƚ�ͨ��;
	colorSpace Ϊ:VK_COLOR_SPACE_SRGB_NONLINEAR_KHR����֧��SRGB��ɫ��
	*/
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
			return{ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };//�����󷵻�
		}

		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}//��������ѡ�������������Ҫ��format
		}

		return availableFormats[0];//ֱ�ӷ��ص�һ��
	}

	/*
	VkPresentModeKHR��ȡ��ֵ��

	VK_PRESENT_MODE_IMMEDIATE_KHR = 0, //��  ��������ʾ�����ܳ���˺��
	VK_PRESENT_MODE_MAILBOX_KHR = 1, // ������
	VK_PRESENT_MODE_FIFO_KHR = 2,  //˫  //�ɱ���˺��
	VK_PRESENT_MODE_FIFO_RELAXED_KHR = 3, //���� FIFO

	
	*/
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;//�������֧��mailbox����ѡ��fifo
	}

	/*
	extent ��Swap Chain��image�ķֱ���(resolution) ,ͨ������window�ĳߴ�һ����
	vulkan������ͨ������currentExtent����width��height��ƥ��window�ķֱ��ʡ�
	������Щwindow Manager�ὫcurrentExtent����Ϊuint32_t�����ֵ������ʾ�����������ò�ͬ��ֵ��
	���ʱ�����ǿ��Դ�minImageExtent��maxImageExtent��ѡ����ƥ��window�ĳߴ�ֵ
	*/
	/*
	Ϊswapѡ����ʵĳߴ�
	��currentExtent.width/height �κ�һ��ֵ����uint32_t�����ֵʱ��ֱ��ѡ��Capabilities.currentExtent, currentExtent�����window�ߴ硣

	���򣬴�Window �ߴ��֧�ֵ����ߴ�(maxImageExtent)ȡ��С�ģ���Ϊ�ߴ粻�ܳ���window��Ҳ���ܳ���capabilities֧�ֵ����ߴ硣
	���Ϊ�˵õ���õ�Ч����ȡ���ֵ��minImageExtent�����ֵ
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

	//����Կ��Ƿ�֧��swapchain  ���֧�ֵ�ϸ�ڣ�Swap Chain�����ܺ����ǵ�window surface������
	/*
	

    Surface ������(Capabilities)(���� : min/max number of images in swap chain, min/max width and height of images)��

    Surface �ĸ�ʽ(formats)(���� : pixel format, color space)��

    ���õ���ʾģʽ(present mode)��

	*/
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);//���VkSurfaceCapabilitiesKHR

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);//���VkSurfaceFormatKHR 

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());//�������֧�ֵ�VkSurfaceFormatKHR��data
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);//���VkPresentModeKHR

		//std::cout << "presentModeCount:" << presentModeCount << std::endl;

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());//�������֧�ֵ�VkPresentModeKHR��data
		}

		return details;
	}
	//����Ƿ���������Ҫ��
	/*
	�Կ�ѡ����߱����¹��ܣ�
	1��֧��geometry shader
	2��֧��ͼ�δ�������
	*/
	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);//��ö���

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		//ֻҪ��һ��format��presentMode���Ǿ���Ϊ���Կ�֧��Swap Chain ���Ҽ���surface

		if (extensionsSupported) {//�Կ�֧�ֵ������¼��swapchain 
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);//��ѯswapchain֧��  ��format��presentmode����Ϊ��ʱ����ʾ������֧��
			//���swapChainAdequate��ֵΪtrue.����Կ϶�Swap Chain�Ѿ���֧���ˡ�����Swap Chain ����������֧��Ҳ����д��swapChainSupport��������
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return indices.isComplete() && extensionsSupported && swapChainAdequate;
	}
	//���֧�ֵ���չ
	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;//������չ��number�洢��extensionCount������
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);//�洢��չ�ľ���ϸ��
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
		//ExtensionProperties �ṩ the name and version of an extension

		//deviceExtensionsΪ�豻֧�ֵ�swapchain��չ��
		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		//�ж�swapchain��չ�Ƿ��Կ�֧��
		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}
		//����Կ��Ƿ�֧��Swap Chain����������ΪSwap Chain�����ܺ����ǵ�window surface�����ݣ����Ի����ѯswapchain��֧��ϸ�ڣ��ȴ���Instance��Logical Device����
		return requiredExtensions.empty();
	}

	//�������
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {//��ҪqueueFamily֧��VK_QUEUE_GRAPHICS_BIT
				indices.graphicsFamily = i;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);//�Ƿ�֧����ʾ

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
	//���extension�б�
	std::vector<const char*> getRequiredExtensions() {
		std::vector<const char*> extensions;

		unsigned int glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);//ʹ��glfw�����

		for (unsigned int i = 0; i < glfwExtensionCount; i++) {
			extensions.push_back(glfwExtensions[i]);
		}

		if (enableValidationLayers) {//��� VK_EXT_DEBUG_REPORT_EXTENSION_NAME
			extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		}

		return extensions;
	}
	//����Ƿ�֧����֤��
	//
	//LunarG validation layers ֻ���ڰ�װ��LunarG SDK��PC�ϲ���ʹ��,�������ǲ���Ҫ���е�layers��LunarG SDK ֻ��ҪVK_LAYER_LUNARG_standard_validation ����
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

	static std::vector<char> readFile(const std::string& filename) {//�����ɫ�����ݴ��뵽td::vector<char>��
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
	//VKAPI_ATTR VkBool32 ȷ�����ܱ�vulkan���õ���Чǩ��
	/*
	��һ��������Ϊ����ֵ
    VK_DEBUG_REPORT_INFORMATION_BIT_EXT
    VK_DEBUG_REPORT_WARNING_BIT_EXT
    VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT
    VK_DEBUG_REPORT_ERROR_BIT_EXT
    VK_DEBUG_REPORT_DEBUG_BIT_EXT

	objType ָ��obj�Ķ����������Ϣ eg�� if obj is a VkPhysicalDevice then objType would be VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT

	Msg ��������������Ҫ��debug��Ϣ��userData������Ҫ����debugCallback������

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
����ģ��ʱ��������tinyobjloader.h
*/
/*
��ȡ����ʱ��������stb
*/
