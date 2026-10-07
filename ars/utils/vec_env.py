import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_fn):
    """
    Worker process for SubprocVecEnv.
    Handles commands from the parent process and executes them on the environment.
    """
    parent_remote.close()
    env = env_fn()
    
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                remote.send((obs, reward, done, info))
            elif cmd == 'reset':
                obs, info = env.reset(**data)
                remote.send((obs, info))
            elif cmd == 'reset_day':
                env.reset_day(**data)
                remote.send(None)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                setattr(env, data[0], data[1])
                remote.send(None)
            elif cmd == 'call_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            else:
                raise NotImplementedError(f"Command {cmd} not implemented")
        except EOFError:
            break

class SubprocVecEnv:
    """
    Vectorized environment that runs multiple environments in parallel using subprocesses.
    """
    def __init__(self, env_fns):
        """
        Args:
            env_fns: List of functions that create environments
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        
        # Start worker processes
        self.ps = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            process = Process(target=worker, args=args, daemon=True)
            process.start()
            self.ps.append(process)
            work_remote.close()
        
        # Get environment attributes from first env
        self.remotes[0].send(('get_attr', 'numAgents'))
        self.numAgents = self.remotes[0].recv()
        
    def step(self, actions):
        """
        Step all environments with the given actions.
        Args:
            actions: List of actions for each environment
        """
        self._assert_not_closed()
        
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
        
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        obs, rewards, dones, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(dones), np.array(infos)
    
    def reset(self, **kwargs):
        """
        Reset all environments.
        """
        self._assert_not_closed()
        
        for remote in self.remotes:
            remote.send(('reset', kwargs))
        
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.array(obs), np.array(infos)
    
    def reset_day(self, day, **kwargs):
        """
        Reset day for all environments.
        """
        self._assert_not_closed()
        
        kwargs['day'] = day
        for remote in self.remotes:
            remote.send(('reset_day', kwargs))
        
        for remote in self.remotes:
            remote.recv()
    
    def close(self):
        """
        Close all environments and terminate processes.
        """
        if self.closed:
            return
        
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        
        for remote in self.remotes:
            remote.send(('close', None))
        
        for p in self.ps:
            p.join()
        
        self.closed = True
    
    def get_attr(self, attr_name, indices=None):
        """
        Get attribute from environments.
        """
        self._assert_not_closed()
        indices = self._get_indices(indices)
        
        for i in indices:
            self.remotes[i].send(('get_attr', attr_name))
        
        return [self.remotes[i].recv() for i in indices]
    
    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute in environments.
        """
        self._assert_not_closed()
        indices = self._get_indices(indices)
        
        for i in indices:
            self.remotes[i].send(('set_attr', (attr_name, value)))
        
        for i in indices:
            self.remotes[i].recv()
    
    def call_method(self, method_name, *args, indices=None, **kwargs):
        """
        Call method on environments.
        """
        self._assert_not_closed()
        indices = self._get_indices(indices)
        
        for i in indices:
            self.remotes[i].send(('call_method', (method_name, args, kwargs)))
        
        return [self.remotes[i].recv() for i in indices]
    
    def _get_indices(self, indices):
        """
        Get indices for environments.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices
    
    def _assert_not_closed(self):
        """
        Assert that environments are not closed.
        """
        assert not self.closed, "Trying to operate on a closed environment"
    
    def __del__(self):
        """
        Destructor to ensure environments are closed.
        """
        if not self.closed:
            self.close()