#FUSE 上conda Jing 环境

import torch
print("GPU 是否可用:", torch.cuda.is_available())
print("当前显卡:", torch.cuda.get_device_name(0))
print("当前显卡:", torch.cuda.get_device_name(0))
import sys
import subprocess
import importlib.metadata
import pkgutil

def check_package(package_name, pip_name=None):
    """检查包是否可用并返回版本信息"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        # 尝试导入包
        module = __import__(package_name)
        # 获取版本信息
        try:
            # 方法1: 直接从模块的 __version__ 属性获取
            if hasattr(module, '__version__'):
                version = module.__version__
            else:
                # 方法2: 尝试从 importlib.metadata 获取
                try:
                    version = importlib.metadata.version(pip_name)
                except:
                    # 方法3: 检查是否有 version 属性
                    if hasattr(module, 'version'):
                        version = module.version
                    else:
                        version = "版本信息未知（可用）"
        except:
            version = "版本信息获取失败（但可用）"
        
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def check_torch_cuda():
    """检查PyTorch CUDA支持"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            return True, f"CUDA {cuda_version}, {device_count} 个设备, GPU: {device_name}"
        else:
            return False, "CUDA不可用 (CPU模式)"
    except Exception as e:
        return False, f"检查CUDA时出错: {str(e)}"

def check_torch_geometric_extras():
    """检查PyTorch Geometric的额外组件"""
    try:
        from torch_geometric import __version__ as tg_version
        
        # 检查必要的扩展
        checks = {}
        
        # 检查是否安装了pyg-lib（可选但推荐）
        try:
            import pyg_lib
            checks['pyg_lib'] = pyg_lib.__version__
        except ImportError:
            checks['pyg_lib'] = "未安装"
        
        # 检查其他可选依赖
        try:
            import torch_cluster
            checks['torch_cluster'] = torch_cluster.__version__
        except ImportError:
            checks['torch_cluster'] = "未安装"
            
        try:
            import torch_scatter
            checks['torch_scatter'] = torch_scatter.__version__
        except ImportError:
            checks['torch_scatter'] = "未安装"
            
        try:
            import torch_sparse
            checks['torch_sparse'] = torch_sparse.__version__
        except ImportError:
            checks['torch_sparse'] = "未安装"
            
        try:
            import torch_spline_conv
            checks['torch_spline_conv'] = torch_spline_conv.__version__
        except ImportError:
            checks['torch_spline_conv'] = "未安装"
        
        return True, checks
    except Exception as e:
        return False, str(e)

def test_basic_functionality():
    """测试基本功能是否正常工作"""
    results = {}
    
    # 测试PyTorch
    try:
        import torch
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        z = x + y
        results['torch_basic'] = True
    except Exception as e:
        results['torch_basic'] = str(e)
    
    # 测试NetworkX
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_edge(1, 2)
        results['networkx_basic'] = True
    except Exception as e:
        results['networkx_basic'] = str(e)
    
    # 测试PyTorch Geometric基本功能
    try:
        from torch_geometric.data import Data
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.tensor([[-1], [0]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index.t().contiguous())
        results['torch_geometric_basic'] = True
    except Exception as e:
        results['torch_geometric_basic'] = str(e)
    
    # 测试sklearn
    try:
        from sklearn.metrics import r2_score
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        score = r2_score(y_true, y_pred)
        results['sklearn_basic'] = True
    except Exception as e:
        results['sklearn_basic'] = str(e)
    
    # 测试tqdm
    try:
        from tqdm import tqdm
        results['tqdm_basic'] = True
    except Exception as e:
        results['tqdm_basic'] = str(e)
    
    return results

def main():
    print("=" * 60)
    print("环境检查报告")
    print("=" * 60)
    
    # Python版本
    print(f"\nPython版本: {sys.version}")
    
    # 检查主要包
    packages = {
        'torch': 'torch',
        'torch.nn': 'torch',  # torch.nn是torch的一部分
        'torch.nn.functional': 'torch',
        'networkx': 'networkx',
        'numpy': 'numpy',
        'torch_geometric': 'torch_geometric',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
    }
    
    print("\n" + "-" * 60)
    print("包检查结果:")
    print("-" * 60)
    
    all_packages_available = True
    
    for package_name, pip_name in packages.items():
        available, info = check_package(package_name, pip_name)
        status = "✓" if available else "✗"
        print(f"{status} {package_name}: {info}")
        if not available:
            all_packages_available = False
    
    # 检查PyTorch CUDA
    print("\n" + "-" * 60)
    print("CUDA检查:")
    print("-" * 60)
    cuda_available, cuda_info = check_torch_cuda()
    status = "✓" if cuda_available else "✗"
    print(f"{status} CUDA: {cuda_info}")
    
    # 检查PyTorch Geometric额外组件
    print("\n" + "-" * 60)
    print("PyTorch Geometric扩展检查:")
    print("-" * 60)
    tg_extras_available, tg_extras_info = check_torch_geometric_extras()
    if tg_extras_available:
        for name, version in tg_extras_info.items():
            if version != "未安装":
                print(f"✓ {name}: {version}")
            else:
                print(f"○ {name}: {version} (可选，不影响基本功能)")
    else:
        print(f"检查扩展时出错: {tg_extras_info}")
    
    # 测试基本功能
    print("\n" + "-" * 60)
    print("基本功能测试:")
    print("-" * 60)
    functional_tests = test_basic_functionality()
    for test_name, result in functional_tests.items():
        if result is True:
            print(f"✓ {test_name}: 正常")
        else:
            print(f"✗ {test_name}: {result}")
            all_packages_available = False
    
    # 最终结论
    print("\n" + "=" * 60)
    if all_packages_available:
        print("✓ 所有必要包都可用！可以继续开发。")
    else:
        print("✗ 有些包不可用，请安装缺失的包。")
    print("=" * 60)
    
    # 如果包缺失，提供安装建议
    if not all_packages_available:
        print("\n安装建议:")
        print("  pip install torch torchvision torchaudio")
        print("  pip install networkx numpy scikit-learn tqdm")
        print("  pip install torch_geometric")
        print("  # 对于PyTorch Geometric的额外优化组件（可选）:")
        print("  pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv")

if __name__ == "__main__":
    main()
