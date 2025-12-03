import { User } from '../types';

const API_BASE_URL = '/api';

export const register = async (email: string, password: string): Promise<User> => {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email, password }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to register');
  }

  const data = await response.json();
  return {
    id: data.user_id,
    email: email,
  };
};

export const login = async (email: string, password: string): Promise<User> => {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email, password }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to login');
  }

  const data = await response.json();
  return {
    id: data.user_id,
    email: data.email,
  };
};

export const getAnalysisHistory = async (userId: number, page: number = 1): Promise<{
  history: any[];
  pagination: {
    total: number;
    pages: number;
    current_page: number;
    per_page: number;
  };
}> => {
  const response = await fetch(`${API_BASE_URL}/analysis/history/${userId}?page=${page}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to fetch analysis history');
  }

  return response.json();
};

export const updateSessionName = async (sessionId: number, name: string, userId?: number): Promise<{ success: boolean; name: string | null }> => {
  const response = await fetch(`${API_BASE_URL}/analysis/session/${sessionId}/name`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name, user_id: userId }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to update session name');
  }

  return response.json();
};