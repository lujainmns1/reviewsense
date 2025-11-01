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